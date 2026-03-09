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


if __name__ == "__main__":
    mcp.run()
