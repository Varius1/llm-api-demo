"""HTTP-клиент для OpenRouter API."""

from __future__ import annotations

import httpx

from .models import (
    OPENROUTER_URL,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    TokenUsage,
)


class OpenRouterClient:
    """Синхронный клиент к OpenRouter (совместим с OpenAI Chat Completions)."""

    def __init__(self, api_key: str, base_url: str = OPENROUTER_URL):
        self.api_key = api_key
        self.base_url = base_url
        self._client = httpx.Client(
            timeout=httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0),
        )

    def close(self) -> None:
        self._client.close()

    def send_raw(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float | None = None,
        transforms: list[str] | None = None,
    ) -> ChatResponse:
        request = ChatRequest(
            model=model,
            messages=messages,
            temperature=temperature,
            transforms=transforms,
        )
        payload = request.model_dump(exclude_none=True)

        response = self._client.post(
            self.base_url,
            json=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text}")

        return ChatResponse.model_validate(response.json())

    def send(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float | None = None,
        transforms: list[str] | None = None,
    ) -> str:
        content, _ = self.send_with_usage(messages, model, temperature, transforms)
        return content

    def send_with_usage(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float | None = None,
        transforms: list[str] | None = None,
    ) -> tuple[str, TokenUsage | None]:
        chat_response = self.send_raw(messages, model, temperature, transforms)

        if chat_response.error is not None:
            raise RuntimeError(f"API ошибка: {chat_response.error.message}")

        if not chat_response.choices:
            raise RuntimeError("Нет ответа в choices")

        return chat_response.choices[0].message.content, chat_response.usage

    def __enter__(self) -> OpenRouterClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
