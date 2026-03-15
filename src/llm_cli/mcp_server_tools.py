"""MCP-сервер #2: Tools & Storage — сохранение файлов, модели, напоминания, мониторинг цен."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("tools-storage")


@mcp.tool()
def save_to_file(content: str, filename: str) -> str:
    """Сохраняет текстовое содержимое в файл.
    Файл создаётся в текущей рабочей директории.
    content — текст для сохранения.
    filename — имя файла (например: report.txt, result.md)."""
    import os
    from pathlib import Path

    safe_name = Path(filename).name
    if not safe_name:
        safe_name = "output.txt"

    allowed_extensions = {".txt", ".md", ".json", ".csv", ".log"}
    suffix = Path(safe_name).suffix.lower()
    if suffix not in allowed_extensions:
        safe_name = safe_name + ".txt"

    output_path = Path(os.getcwd()) / safe_name

    try:
        output_path.write_text(content, encoding="utf-8")
        size_bytes = output_path.stat().st_size
        size_label = f"{size_bytes} байт" if size_bytes < 1024 else f"{size_bytes / 1024:.1f} КБ"
        return (
            f"Файл сохранён успешно!\n"
            f"  Путь:   {output_path}\n"
            f"  Размер: {size_label}\n"
            f"  Строк:  {content.count(chr(10)) + 1}"
        )
    except OSError as e:
        return f"Ошибка сохранения файла '{safe_name}': {e}"


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


if __name__ == "__main__":
    mcp.run()
