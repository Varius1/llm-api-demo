"""HTTP-клиент для OpenRouter API."""

from __future__ import annotations

import time

import httpx

from .models import (
    OPENROUTER_URL,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    TokenUsage,
    ToolDefinition,
)


def _merge_system_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Объединить все system-сообщения в одно первое.

    Некоторые локальные модели (например Qwen3 через llama.cpp) требуют
    строго одного system-сообщения и строго в начале списка.
    """
    system_parts: list[str] = []
    non_system: list[ChatMessage] = []
    for msg in messages:
        if msg.role == "system":
            if msg.content:
                system_parts.append(msg.content)
        else:
            non_system.append(msg)

    if not system_parts:
        return non_system

    merged = ChatMessage(role="system", content="\n\n".join(system_parts))
    return [merged, *non_system]


class OpenRouterClient:
    """Синхронный клиент к OpenRouter (совместим с OpenAI Chat Completions)."""

    def __init__(self, api_key: str, base_url: str = OPENROUTER_URL):
        self.api_key = api_key
        self.base_url = base_url
        self._max_retries = 3
        self._client = httpx.Client(
            timeout=httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0),
        )

    @property
    def is_local(self) -> bool:
        return self.base_url != OPENROUTER_URL

    def close(self) -> None:
        self._client.close()

    def send_raw(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float | None = None,
        transforms: list[str] | None = None,
        tools: list[ToolDefinition] | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        if self.is_local:
            messages = _merge_system_messages(messages)

        request = ChatRequest(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            transforms=transforms,
            tools=tools,
        )
        payload = request.model_dump(exclude_none=True)

        response: httpx.Response | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.post(
                    self.base_url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                )
                break
            except (httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectError) as exc:
                if attempt >= self._max_retries:
                    raise RuntimeError(
                        "Сетевой сбой при обращении к OpenRouter после повторных попыток. "
                        "Проверьте соединение и повторите запрос."
                    ) from exc
                # Короткий backoff: 0.5s, 1.0s, 1.5s
                time.sleep(0.5 * (attempt + 1))

        assert response is not None

        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text}")

        return ChatResponse.model_validate(response.json())

    def send(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float | None = None,
        transforms: list[str] | None = None,
        tools: list[ToolDefinition] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        content, _ = self.send_with_usage(messages, model, temperature, transforms, tools=tools, max_tokens=max_tokens)
        return content

    def send_with_usage(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float | None = None,
        transforms: list[str] | None = None,
        tools: list[ToolDefinition] | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, TokenUsage | None]:
        chat_response = self.send_raw(messages, model, temperature, transforms, tools=tools, max_tokens=max_tokens)

        if chat_response.error is not None:
            raise RuntimeError(f"API ошибка: {chat_response.error.message}")

        if not chat_response.choices:
            raise RuntimeError("Нет ответа в choices")

        choice = chat_response.choices[0]
        content = choice.message.content or ""
        return content, chat_response.usage

    def __enter__(self) -> OpenRouterClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
