"""Локальный MCP-сервер с демонстрационными инструментами (stdio-транспорт)."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("llm-cli-tools")


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
