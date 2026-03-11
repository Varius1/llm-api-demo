"""Планировщик: хранилище напоминаний и истории цен + фоновый демон-коллектор."""

from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from platformdirs import user_config_dir
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich import box

APP_NAME = "llm-cli"
STORAGE_FILENAME = "scheduler_storage.json"

_COINGECKO_IDS: dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "BNB": "binancecoin",
    "XRP": "ripple",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "TON": "the-open-network",
}

# Максимальное количество точек истории на символ
MAX_HISTORY_POINTS = 500

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Storage helpers
# ─────────────────────────────────────────────────────────────────────────────

def _storage_path() -> Path:
    return Path(user_config_dir(APP_NAME, appauthor=False, ensure_exists=True)) / STORAGE_FILENAME


def _load_storage() -> dict[str, Any]:
    path = _storage_path()
    if not path.exists():
        return _empty_storage()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return _empty_storage()
        if "reminders" not in data:
            data["reminders"] = []
        if "price_history" not in data:
            data["price_history"] = {}
        if "monitor_config" not in data:
            data["monitor_config"] = {"symbols": [], "interval_seconds": 30}
        return data
    except (OSError, json.JSONDecodeError):
        return _empty_storage()


def _save_storage(data: dict[str, Any]) -> None:
    path = _storage_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _empty_storage() -> dict[str, Any]:
    return {
        "reminders": [],
        "price_history": {},
        "monitor_config": {"symbols": [], "interval_seconds": 30},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public storage API (используется из mcp_server.py)
# ─────────────────────────────────────────────────────────────────────────────

def add_reminder(message: str, delay_seconds: int) -> dict[str, Any]:
    """Добавить напоминание, которое сработает через delay_seconds секунд."""
    data = _load_storage()
    fire_at = time.time() + delay_seconds
    reminder: dict[str, Any] = {
        "id": str(uuid.uuid4())[:8],
        "message": message,
        "fire_at": fire_at,
        "fire_at_human": datetime.fromtimestamp(fire_at).strftime("%H:%M:%S"),
        "fired": False,
        "created_at": time.time(),
    }
    data["reminders"].append(reminder)
    _save_storage(data)
    return reminder


def get_reminders() -> list[dict[str, Any]]:
    """Вернуть все напоминания (и сработавшие, и ожидающие)."""
    data = _load_storage()
    now = time.time()
    changed = False
    for r in data["reminders"]:
        if not r.get("fired") and r.get("fire_at", 0) <= now:
            r["fired"] = True
            r["fired_at_human"] = datetime.now().strftime("%H:%M:%S")
            changed = True
    if changed:
        _save_storage(data)
    return data["reminders"]


def set_monitor_config(symbols: list[str], interval_seconds: int) -> dict[str, Any]:
    """Обновить конфиг периодического мониторинга цен."""
    data = _load_storage()
    valid = [s.upper() for s in symbols if s.upper() in _COINGECKO_IDS]
    data["monitor_config"] = {
        "symbols": valid,
        "interval_seconds": max(5, interval_seconds),
    }
    _save_storage(data)
    return data["monitor_config"]


def get_price_history(symbol: str) -> list[dict[str, Any]]:
    """Вернуть историю цен для символа."""
    data = _load_storage()
    return data["price_history"].get(symbol.upper(), [])


def append_price_point(symbol: str, price: float) -> None:
    """Добавить точку истории цен."""
    data = _load_storage()
    sym = symbol.upper()
    if sym not in data["price_history"]:
        data["price_history"][sym] = []
    data["price_history"][sym].append({
        "ts": time.time(),
        "ts_human": datetime.now().strftime("%H:%M:%S"),
        "price": price,
    })
    # Обрезаем до максимума
    if len(data["price_history"][sym]) > MAX_HISTORY_POINTS:
        data["price_history"][sym] = data["price_history"][sym][-MAX_HISTORY_POINTS:]
    _save_storage(data)


def get_price_summary(symbol: str, last_n_minutes: int = 5) -> dict[str, Any]:
    """Агрегированная сводка: мин/макс/средняя цена, тренд, кол-во точек."""
    history = get_price_history(symbol)
    if not history:
        return {"symbol": symbol.upper(), "error": "Нет данных. Запустите мониторинг командой start_price_monitor."}

    cutoff = time.time() - last_n_minutes * 60
    points = [p for p in history if p.get("ts", 0) >= cutoff]

    if not points:
        return {
            "symbol": symbol.upper(),
            "error": f"Нет данных за последние {last_n_minutes} мин. "
                     f"Всего точек в истории: {len(history)}. "
                     f"Последняя: {history[-1]['ts_human']} — ${history[-1]['price']:,.2f}",
        }

    prices = [p["price"] for p in points]
    first_price = points[0]["price"]
    last_price = points[-1]["price"]
    delta = last_price - first_price
    delta_pct = (delta / first_price * 100) if first_price else 0
    trend = "↑" if delta > 0 else ("↓" if delta < 0 else "→")

    return {
        "symbol": symbol.upper(),
        "points_count": len(points),
        "period_minutes": last_n_minutes,
        "min_price": min(prices),
        "max_price": max(prices),
        "avg_price": sum(prices) / len(prices),
        "first_price": first_price,
        "last_price": last_price,
        "delta_usd": delta,
        "delta_pct": round(delta_pct, 3),
        "trend": trend,
        "first_time": points[0]["ts_human"],
        "last_time": points[-1]["ts_human"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# CoinGecko fetch
# ─────────────────────────────────────────────────────────────────────────────

def fetch_price(symbol: str) -> float | None:
    coin_id = _COINGECKO_IDS.get(symbol.upper())
    if not coin_id:
        return None
    try:
        resp = httpx.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": coin_id, "vs_currencies": "usd"},
            timeout=10,
        )
        resp.raise_for_status()
        return float(resp.json()[coin_id]["usd"])
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Background collector thread
# ─────────────────────────────────────────────────────────────────────────────

class PriceCollector(threading.Thread):
    """Фоновый поток: периодически собирает цены и помечает сработавшие напоминания."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        last_collect_time: float = 0.0

        while not self._stop_event.is_set():
            data = _load_storage()
            cfg = data.get("monitor_config", {})
            symbols: list[str] = cfg.get("symbols", [])
            interval: int = int(cfg.get("interval_seconds", 30))

            now = time.time()

            # Пометить сработавшие напоминания
            changed = False
            for r in data.get("reminders", []):
                if not r.get("fired") and r.get("fire_at", 0) <= now:
                    r["fired"] = True
                    r["fired_at_human"] = datetime.now().strftime("%H:%M:%S")
                    changed = True
                    console.print(
                        f"\n  [bold yellow]🔔 Напоминание сработало:[/bold yellow] "
                        f"[white]{r['message']}[/white] "
                        f"[dim]({r['fired_at_human']})[/dim]\n"
                    )
            if changed:
                _save_storage(data)

            # Сбор цен по расписанию
            if symbols and (now - last_collect_time) >= interval:
                last_collect_time = now
                ts_str = datetime.now().strftime("%H:%M:%S")
                parts: list[str] = []
                for sym in symbols:
                    price = fetch_price(sym)
                    if price is not None:
                        append_price_point(sym, price)
                        parts.append(f"[bold cyan]{sym}[/bold cyan]: [green]${price:,.2f}[/green]")
                    else:
                        parts.append(f"[bold cyan]{sym}[/bold cyan]: [red]ошибка[/red]")
                console.print(
                    f"  [dim]{ts_str}[/dim]  ⏱  " + "  ".join(parts)
                )

            self._stop_event.wait(timeout=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Run daemon (CLI --scheduler)
# ─────────────────────────────────────────────────────────────────────────────

def run_scheduler_daemon() -> None:
    """Запустить планировщик в режиме 24/7 (блокирующий, прерывается Ctrl+C)."""
    console.print(Panel(
        "[bold cyan]Планировщик запущен[/bold cyan]\n"
        "[dim]Сбор цен + проверка напоминаний каждую секунду[/dim]\n"
        "[dim]Конфиг мониторинга задаётся через MCP: start_price_monitor()[/dim]\n"
        "[dim]Ctrl+C для остановки[/dim]",
        border_style="cyan",
        expand=False,
    ))
    console.print()

    data = _load_storage()
    cfg = data.get("monitor_config", {})
    symbols = cfg.get("symbols", [])
    interval = cfg.get("interval_seconds", 30)

    if symbols:
        console.print(
            f"[green]Мониторинг:[/green] {', '.join(symbols)} "
            f"[dim]каждые {interval} сек[/dim]\n"
        )
    else:
        console.print("[yellow]Символы не настроены. Используйте llm-cli --tools и start_price_monitor()[/yellow]\n")

    collector = PriceCollector()
    collector.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        collector.stop()
        console.print("\n[dim]Планировщик остановлен.[/dim]")


# ─────────────────────────────────────────────────────────────────────────────
# Auto-demo (CLI --scheduler-demo)
# ─────────────────────────────────────────────────────────────────────────────

_DEMO_MODEL = "openai/gpt-4o-mini"

_DEMO_STEPS = [
    (
        "Установи напоминание с текстом 'Проверить серверы' через 45 секунд.",
        "add_reminder",
    ),
    (
        "Запусти мониторинг цен для BTC, ETH и SOL с интервалом 15 секунд.",
        "start_price_monitor",
    ),
    (
        "Покажи агрегированную сводку цен BTC за последние 2 минуты.",
        "get_price_summary",
    ),
    (
        "Покажи все напоминания — есть ли сработавшие?",
        "get_pending_reminders",
    ),
]


async def _run_demo_step(
    step_num: int,
    prompt: str,
    tool_hint: str,
    agent: Any,
) -> str:
    console.print()
    console.print(Rule(
        f"[bold magenta]Шаг {step_num}/4 — {tool_hint}[/bold magenta]",
        style="magenta",
    ))
    console.print()
    console.print(Panel(
        f"[dim]Промпт агенту:[/dim]\n[white]{prompt}[/white]",
        border_style="dim",
        expand=False,
    ))
    console.print()

    reply = await agent.run_async(prompt)

    console.print()
    console.print(Panel(
        reply,
        title="[bold green]Ответ LLM[/bold green]",
        border_style="green",
        expand=False,
    ))
    return reply


async def _run_demo(api_key: str) -> None:
    from .agent import Agent
    from .api import OpenRouterClient
    from .mcp_client import MCPSession

    console.print(Panel(
        "[bold cyan]Scheduler Demo — Умный мониторинг крипто-портфеля[/bold cyan]\n"
        "[dim]Агент сам вызовет MCP-инструменты: напоминание, мониторинг, сводка[/dim]",
        border_style="cyan",
        expand=False,
    ))
    console.print()

    # Запустить фоновый коллектор
    collector = PriceCollector()
    collector.start()
    console.print("[green]✓[/green] Фоновый коллектор запущен\n")

    async with MCPSession() as mcp:
        tools = mcp.get_tools_schema()

        table = Table(
            title=f"MCP-инструменты планировщика ({len(tools)})",
            box=box.ROUNDED,
            border_style="cyan",
            header_style="bold magenta",
        )
        table.add_column("Инструмент", style="bold yellow", no_wrap=True)
        table.add_column("Описание", style="white")
        for t in tools:
            if t.function.name in {
                "add_reminder", "get_pending_reminders",
                "start_price_monitor", "get_price_summary",
            }:
                table.add_row(t.function.name, t.function.description)
        console.print(table)
        console.print(f"[dim]Модель: {_DEMO_MODEL}[/dim]\n")

        with OpenRouterClient(api_key) as client:
            agent = Agent(client=client, model=_DEMO_MODEL, mcp_session=mcp)

            # Шаг 1: напоминание
            await _run_demo_step(1, _DEMO_STEPS[0][0], _DEMO_STEPS[0][1], agent)

            # Шаг 2: запуск мониторинга
            await _run_demo_step(2, _DEMO_STEPS[1][0], _DEMO_STEPS[1][1], agent)

            # Пауза — ждём пока накопятся данные и сработает напоминание
            wait_seconds = 50
            console.print()
            console.print(Rule("[bold yellow]Ожидание накопления данных...[/bold yellow]", style="yellow"))
            console.print(f"[dim]Ждём {wait_seconds} сек: коллектор собирает цены, напоминание срабатывает в ~45 сек[/dim]\n")

            for remaining in range(wait_seconds, 0, -5):
                time.sleep(5)
                console.print(f"  [dim]⏳ осталось ~{remaining - 5} сек...[/dim]")

            console.print()

            # Шаг 3: сводка по BTC
            await _run_demo_step(3, _DEMO_STEPS[2][0], _DEMO_STEPS[2][1], agent)

            # Шаг 4: список напоминаний
            await _run_demo_step(4, _DEMO_STEPS[3][0], _DEMO_STEPS[3][1], agent)

    collector.stop()

    console.print()
    console.print(Rule("[bold cyan]Демо завершено[/bold cyan]", style="cyan"))
    console.print()
    storage_path = _storage_path()
    console.print(f"[dim]Данные сохранены в: {storage_path}[/dim]")


def run_scheduler_demo(api_key: str) -> None:
    """Полностью автоматическое демо планировщика с LLM-агентом."""
    import asyncio
    asyncio.run(_run_demo(api_key))
