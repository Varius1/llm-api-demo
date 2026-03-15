"""MCP-сервер #1: Data & Analytics — поиск, погода, крипто, вычисления, суммаризация."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("data-analytics")

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
def get_crypto_price(symbol: str) -> str:
    """Возвращает текущую цену криптовалюты в USD.
    Поддерживаемые символы: BTC, ETH, SOL, BNB, XRP, ADA, DOGE, TON."""
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


if __name__ == "__main__":
    mcp.run()
