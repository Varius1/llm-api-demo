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


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline инструменты: search → summarize → save_to_file
# ─────────────────────────────────────────────────────────────────────────────

_KNOWLEDGE_BASE: dict[str, list[str]] = {
    "python": [
        "Python — интерпретируемый высокоуровневый язык программирования общего назначения, созданный Гвидо ван Россумом в 1991 году.",
        "Python поддерживает несколько парадигм: объектно-ориентированное, императивное, функциональное и процедурное программирование.",
        "Python использует динамическую типизацию и автоматическое управление памятью через сборщик мусора.",
        "Стандартная библиотека Python включает более 200 модулей: работа с файлами, сетью, JSON, регулярными выражениями и многим другим.",
        "Python широко применяется в науке о данных, машинном обучении, веб-разработке (Django, FastAPI) и автоматизации.",
        "Синтаксис Python основан на отступах — блоки кода выделяются пробелами, а не скобками.",
        "Python 3.0 был выпущен в 2008 году и не обратно совместим с Python 2.",
        "Популярные фреймворки: NumPy, Pandas, TensorFlow, PyTorch, Flask, Django, FastAPI.",
    ],
    "mcp": [
        "MCP (Model Context Protocol) — открытый протокол для интеграции языковых моделей с внешними инструментами и данными.",
        "MCP использует архитектуру клиент-сервер: сервер предоставляет инструменты, клиент (LLM-агент) их вызывает.",
        "Транспорт MCP: stdio (локальный процесс) и SSE (HTTP). Stdio подходит для CLI-приложений.",
        "MCP-инструменты описываются JSON-схемой: имя, описание и параметры с типами.",
        "LLM вызывает MCP-инструменты через механизм tool calling: модель возвращает tool_calls, клиент выполняет их и возвращает результаты.",
        "FastMCP — Python-библиотека для быстрого создания MCP-серверов с декоратором @mcp.tool().",
    ],
    "llm": [
        "LLM (Large Language Model) — большая языковая модель, обученная на огромных объёмах текста для понимания и генерации языка.",
        "Популярные LLM: GPT-4, Claude 3.5, Gemini 1.5, Llama 3 — различаются размером, скоростью и ценой.",
        "Температура (temperature) — параметр генерации: 0 = детерминированный вывод, 2 = максимальная случайность.",
        "Token — минимальная единица текста для LLM, примерно 4 символа или 0.75 слова в английском.",
        "Context window — максимальный размер входного контекста модели, обычно от 8k до 200k токенов.",
        "Tool calling (function calling) позволяет LLM вызывать внешние функции, получая результат обратно в контекст.",
    ],
    "ai": [
        "Искусственный интеллект (ИИ) — раздел информатики, изучающий создание систем, выполняющих задачи, требующие человеческого интеллекта.",
        "Машинное обучение — подраздел ИИ: алгоритмы обучаются на данных без явного программирования.",
        "Deep Learning (глубокое обучение) использует многослойные нейронные сети для решения сложных задач.",
        "Генеративный ИИ создаёт новый контент: текст, изображения, код, аудио — на основе обученных паттернов.",
        "RAG (Retrieval-Augmented Generation) — архитектура, сочетающая поиск в базе знаний с генерацией LLM.",
    ],
}

_DEFAULT_FACTS = [
    "Запрос выполнен через встроенную базу знаний llm-cli.",
    "Результаты отсортированы по релевантности.",
    "Данные актуальны на момент сборки пакета.",
]


@mcp.tool()
def search(query: str, max_results: int = 5) -> str:
    """Поиск информации по запросу во встроенной базе знаний.
    Возвращает список фактов и статей, связанных с запросом.
    query — поисковый запрос (ключевые слова).
    max_results — максимальное количество результатов (по умолчанию 5)."""
    query_lower = query.lower()
    matched: list[str] = []

    for topic, facts in _KNOWLEDGE_BASE.items():
        if topic in query_lower or any(word in query_lower for word in topic.split()):
            matched.extend(facts)

    if not matched:
        for topic, facts in _KNOWLEDGE_BASE.items():
            for fact in facts:
                if any(word in fact.lower() for word in query_lower.split() if len(word) > 3):
                    matched.append(fact)

    if not matched:
        matched = _DEFAULT_FACTS[:]

    matched = list(dict.fromkeys(matched))
    matched = matched[:max(1, max_results)]

    lines = [f"Результаты поиска по запросу: «{query}» ({len(matched)} результатов)\n"]
    for i, fact in enumerate(matched, 1):
        lines.append(f"{i}. {fact}")
    return "\n".join(lines)


@mcp.tool()
def summarize(text: str, max_sentences: int = 3) -> str:
    """Сжимает переданный текст до краткого резюме из max_sentences предложений.
    Использует локальную эвристику без обращения к LLM.
    text — исходный текст для суммаризации.
    max_sentences — максимальное количество предложений в резюме (по умолчанию 3)."""
    import re

    if not text or not text.strip():
        return "Пустой текст — суммаризация невозможна."

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    numbered = re.compile(r"^\d+\.\s+")
    content_lines = [numbered.sub("", line) for line in lines if not line.startswith("Результаты поиска")]

    sentences: list[str] = []
    for line in content_lines:
        parts = re.split(r"(?<=[.!?])\s+", line)
        sentences.extend(p.strip() for p in parts if p.strip() and len(p.strip()) > 15)

    if not sentences:
        return text[:300] + ("..." if len(text) > 300 else "")

    n = max(1, max_sentences)
    selected = sentences[:n]

    summary = " ".join(selected)
    if not summary.endswith((".", "!", "?")):
        summary += "."

    return f"Резюме ({len(selected)} предл. из {len(sentences)} найденных):\n{summary}"


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
def get_git_info() -> str:
    """Возвращает информацию о текущем состоянии git-репозитория: ветка, последний коммит, статус."""
    import subprocess

    results: dict[str, str] = {}

    try:
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        results["branch"] = branch or "(detached HEAD)"
    except subprocess.CalledProcessError:
        results["branch"] = "неизвестно (не git-репозиторий?)"

    try:
        last_commit = subprocess.check_output(
            ["git", "log", "-1", "--oneline"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        results["last_commit"] = last_commit or "нет коммитов"
    except subprocess.CalledProcessError:
        results["last_commit"] = "нет коммитов"

    try:
        status_out = subprocess.check_output(
            ["git", "status", "--short"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        changed_files = len(status_out.splitlines()) if status_out else 0
        results["changed_files"] = str(changed_files)
    except subprocess.CalledProcessError:
        results["changed_files"] = "0"

    return (
        f"Ветка:           {results['branch']}\n"
        f"Последний коммит: {results['last_commit']}\n"
        f"Изменённых файлов: {results['changed_files']}"
    )


if __name__ == "__main__":
    mcp.run()
