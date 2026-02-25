"""Pydantic-модели для OpenRouter API и бенчмарка."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    transforms: list[str] | None = None


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatTurnStats(BaseModel):
    request_tokens_estimated: int
    history_tokens_estimated: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    turn_cost_usd: float | None = None
    session_prompt_tokens: int = 0
    session_completion_tokens: int = 0
    session_total_tokens: int = 0
    session_cost_usd: float = 0.0


class ChatError(BaseModel):
    message: str = "Unknown error"
    code: int | None = None


class ChatChoice(BaseModel):
    message: ChatMessage


class ChatResponse(BaseModel):
    choices: list[ChatChoice] = Field(default_factory=list)
    error: ChatError | None = None
    usage: TokenUsage | None = None
    model: str | None = None


class ModelConfig(BaseModel):
    id: str
    tier: str
    display_name: str
    input_price_per_million: float
    output_price_per_million: float
    url: str = ""

    def format_url(self) -> str:
        return self.url or f"https://openrouter.ai/{self.id}"


class BenchmarkResult(BaseModel):
    model: ModelConfig
    response: str = ""
    usage: TokenUsage | None = None
    duration_ms: int = 0
    cost_usd: float = 0.0
    error: str | None = None

    @staticmethod
    def calculate_cost(usage: TokenUsage | None, model: ModelConfig) -> float:
        if usage is None:
            return 0.0
        input_cost = usage.prompt_tokens / 1_000_000 * model.input_price_per_million
        output_cost = usage.completion_tokens / 1_000_000 * model.output_price_per_million
        return input_cost + output_cost


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_MODEL = "google/gemma-2-9b-it"

DEFAULT_MODEL_INPUT_PRICE_PER_MILLION = 0.03
DEFAULT_MODEL_OUTPUT_PRICE_PER_MILLION = 0.09

BENCHMARK_MODELS = [
    ModelConfig(
        id="meta-llama/llama-3.3-70b-instruct",
        tier="Слабая (дешёвая)",
        display_name="Llama 3.3 70B",
        input_price_per_million=0.10,
        output_price_per_million=0.32,
        url="https://openrouter.ai/meta-llama/llama-3.3-70b-instruct",
    ),
    ModelConfig(
        id="google/gemini-2.5-flash",
        tier="Средняя",
        display_name="Gemini 2.5 Flash",
        input_price_per_million=0.30,
        output_price_per_million=2.50,
        url="https://openrouter.ai/google/gemini-2.5-flash",
    ),
    ModelConfig(
        id="anthropic/claude-sonnet-4",
        tier="Сильная (дорогая)",
        display_name="Claude Sonnet 4",
        input_price_per_million=3.00,
        output_price_per_million=15.00,
        url="https://openrouter.ai/anthropic/claude-sonnet-4",
    ),
]

BENCHMARK_PROMPT = (
    "Объясни принцип работы квантового компьютера и в чём его отличие "
    "от классического. Ответ дай в 3-5 предложениях."
)

JUDGE_MODEL = ModelConfig(
    id="google/gemini-2.5-flash",
    tier="Судья",
    display_name="Gemini 2.5 Flash",
    input_price_per_million=0.30,
    output_price_per_million=2.50,
    url="https://openrouter.ai/google/gemini-2.5-flash",
)


def get_model_token_prices_per_million(model_id: str) -> tuple[float, float] | None:
    if model_id == DEFAULT_MODEL:
        return (DEFAULT_MODEL_INPUT_PRICE_PER_MILLION, DEFAULT_MODEL_OUTPUT_PRICE_PER_MILLION)

    for model in BENCHMARK_MODELS:
        if model.id == model_id:
            return (model.input_price_per_million, model.output_price_per_million)

    return None


def calculate_usage_cost_usd(usage: TokenUsage | None, model_id: str) -> float | None:
    if usage is None:
        return None

    prices = get_model_token_prices_per_million(model_id)
    if prices is None:
        return None

    input_price, output_price = prices
    input_cost = usage.prompt_tokens / 1_000_000 * input_price
    output_cost = usage.completion_tokens / 1_000_000 * output_price
    return input_cost + output_cost
