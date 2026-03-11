"""Локальный MCP-сервер с демонстрационными инструментами (stdio-транспорт)."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("llm-cli-tools")


# ─────────────────────────────────────────────────────────────────────────────
# Планировщик — напоминания и мониторинг цен
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def add_reminder(message: str, delay_seconds: int) -> str:
    """Добавить напоминание, которое сработает через delay_seconds секунд.
    Данные сохраняются в JSON-хранилище и проверяются фоновым планировщиком."""
    from .scheduler_daemon import add_reminder as _add
    r = _add(message, delay_seconds)
    return (
        f"Напоминание установлено!\n"
        f"  ID: {r['id']}\n"
        f"  Текст: {r['message']}\n"
        f"  Сработает в: {r['fire_at_human']} (через {delay_seconds} сек)"
    )


@mcp.tool()
def get_pending_reminders() -> str:
    """Вернуть список всех напоминаний: сработавшие и ожидающие."""
    from .scheduler_daemon import get_reminders
    reminders = get_reminders()
    if not reminders:
        return "Напоминаний нет."

    lines = []
    fired = [r for r in reminders if r.get("fired")]
    pending = [r for r in reminders if not r.get("fired")]

    if fired:
        lines.append(f"✅ Сработавшие ({len(fired)}):")
        for r in fired:
            lines.append(f"  [{r['id']}] {r['message']} — сработало в {r.get('fired_at_human', '?')}")
    if pending:
        lines.append(f"⏳ Ожидающие ({len(pending)}):")
        for r in pending:
            lines.append(f"  [{r['id']}] {r['message']} — сработает в {r['fire_at_human']}")

    return "\n".join(lines)


@mcp.tool()
def start_price_monitor(symbols: str, interval_seconds: int = 15) -> str:
    """Запустить периодический сбор цен криптовалют.
    symbols — символы через запятую, например: BTC,ETH,SOL
    interval_seconds — интервал сбора в секундах (минимум 5).
    Данные сохраняются в JSON, коллектор читает конфиг при каждом тике."""
    from .scheduler_daemon import set_monitor_config
    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
    cfg = set_monitor_config(symbol_list, interval_seconds)
    if not cfg["symbols"]:
        return (
            f"Ни один символ не распознан из: {symbols}\n"
            f"Поддерживаются: BTC, ETH, SOL, BNB, XRP, ADA, DOGE, TON"
        )
    return (
        f"Мониторинг настроен!\n"
        f"  Символы: {', '.join(cfg['symbols'])}\n"
        f"  Интервал: {cfg['interval_seconds']} сек\n"
        f"  Данные пишутся в scheduler_storage.json\n"
        f"  Фоновый коллектор подхватит конфиг автоматически."
    )


@mcp.tool()
def get_price_summary(symbol: str, last_n_minutes: int = 5) -> str:
    """Агрегированная сводка цен криптовалюты за последние last_n_minutes минут.
    Включает мин/макс/среднюю цену, тренд (↑↓→) и дельту.
    symbol — символ: BTC, ETH, SOL и т.д."""
    from .scheduler_daemon import get_price_summary as _summary
    s = _summary(symbol, last_n_minutes)
    if "error" in s:
        return s["error"]
    trend_label = {"↑": "Рост", "↓": "Падение", "→": "Стабильно"}.get(s["trend"], "")
    sign = "+" if s["delta_usd"] >= 0 else ""
    return (
        f"{s['symbol']} — сводка за {s['period_minutes']} мин "
        f"({s['points_count']} точек, {s['first_time']}–{s['last_time']})\n"
        f"  Мин:    ${s['min_price']:,.2f}\n"
        f"  Макс:   ${s['max_price']:,.2f}\n"
        f"  Средняя: ${s['avg_price']:,.2f}\n"
        f"  Первая: ${s['first_price']:,.2f}  →  Последняя: ${s['last_price']:,.2f}\n"
        f"  Тренд:  {s['trend']} {trend_label} "
        f"({sign}{s['delta_usd']:,.2f} USD, {sign}{s['delta_pct']}%)"
    )


@mcp.tool()
def get_weather(city: str) -> str:
    """Возвращает текущую погоду для указанного города."""
    mock_data: dict[str, str] = {
        "москва": "Пасмурно, -3°C, ветер 5 м/с",
        "london": "Cloudy, 8°C, wind 12 km/h",
        "new york": "Sunny, 15°C, wind 8 km/h",
    }
    return mock_data.get(city.lower(), f"Нет данных для города '{city}'")


@mcp.tool()
def calculate(expression: str) -> str:
    """Вычисляет математическое выражение и возвращает результат."""
    try:
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "Ошибка: недопустимые символы в выражении"
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Ошибка вычисления: {e}"


@mcp.tool()
def list_models() -> str:
    """Возвращает список LLM-моделей, доступных по умолчанию в llm-cli."""
    models = [
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "anthropic/claude-3.5-sonnet",
        "google/gemini-flash-1.5",
        "meta-llama/llama-3.1-8b-instruct",
    ]
    return "\n".join(f"• {m}" for m in models)


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


@mcp.tool()
def get_crypto_price(symbol: str) -> str:
    """Возвращает текущую цену криптовалюты в USD. Поддерживаемые символы: BTC, ETH, SOL, BNB, XRP, ADA, DOGE, TON."""
    import httpx

    coin_id = _COINGECKO_IDS.get(symbol.upper())
    if not coin_id:
        return f"Неизвестный символ '{symbol}'. Поддерживаются: {', '.join(_COINGECKO_IDS)}"

    url = "https://api.coingecko.com/api/v3/simple/price"
    try:
        resp = httpx.get(url, params={"ids": coin_id, "vs_currencies": "usd"}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        price = data[coin_id]["usd"]
        return f"{symbol.upper()}: ${price:,} USD"
    except httpx.HTTPStatusError as e:
        return f"Ошибка: не удалось получить цену для {symbol.upper()} (HTTP {e.response.status_code})"
    except Exception as e:
        return f"Ошибка запроса к CoinGecko API: {e}"


if __name__ == "__main__":
    mcp.run()
