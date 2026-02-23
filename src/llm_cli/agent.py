"""LLM-агент — инкапсулирует логику диалога с моделью."""

from __future__ import annotations

from .api import OpenRouterClient
from .models import ChatMessage


class Agent:
    """Агент-посредник между пользователем и LLM.

    Хранит историю диалога и управляет параметрами генерации,
    предоставляя единый метод ``run()`` для взаимодействия.
    """

    def __init__(
        self,
        client: OpenRouterClient,
        model: str,
        temperature: float | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._history: list[ChatMessage] = []

        if system_prompt:
            self._history.append(ChatMessage(role="system", content=system_prompt))

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        self._model = value

    @property
    def temperature(self) -> float | None:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float | None) -> None:
        self._temperature = value

    @property
    def history(self) -> list[ChatMessage]:
        return list(self._history)

    def run(self, user_input: str) -> str:
        """Отправить сообщение пользователя в LLM и вернуть ответ.

        Автоматически добавляет сообщение и ответ в историю,
        так что следующий вызов ``run()`` будет содержать весь контекст.
        """
        self._history.append(ChatMessage(role="user", content=user_input))

        reply = self._client.send(
            self._history,
            self._model,
            self._temperature,
        )

        self._history.append(ChatMessage(role="assistant", content=reply))
        return reply

    def clear_history(self) -> None:
        """Очистить историю, сохранив системный промпт (если был)."""
        system = [m for m in self._history if m.role == "system"]
        self._history = system
