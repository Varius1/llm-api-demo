"""LLM-агент — инкапсулирует логику диалога с моделью."""

from __future__ import annotations

import json
from pathlib import Path

from platformdirs import user_config_dir

from .api import OpenRouterClient
from .models import ChatMessage

APP_NAME = "llm-cli"
HISTORY_FILENAME = "history.json"


def _history_path() -> Path:
    config_dir = Path(user_config_dir(APP_NAME, appauthor=False, ensure_exists=True))
    return config_dir / HISTORY_FILENAME


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
        self._history_file = _history_path()
        self._history = self._load_history()
        self._restored_messages_count = len(self._history)

        if system_prompt:
            has_system = any(msg.role == "system" for msg in self._history)
            if not has_system:
                self._history.insert(0, ChatMessage(role="system", content=system_prompt))
                self._save_history()

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

    @property
    def restored_messages_count(self) -> int:
        return self._restored_messages_count

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
        self._save_history()
        return reply

    def clear_history(self) -> None:
        """Очистить историю, сохранив системный промпт (если был)."""
        system = [m for m in self._history if m.role == "system"]
        self._history = system
        self._save_history()

    def _load_history(self) -> list[ChatMessage]:
        if not self._history_file.exists():
            return []

        try:
            raw_text = self._history_file.read_text(encoding="utf-8")
            if not raw_text.strip():
                return []
            payload = json.loads(raw_text)
            if not isinstance(payload, list):
                return []
        except (OSError, json.JSONDecodeError):
            return []

        history: list[ChatMessage] = []
        for item in payload:
            try:
                history.append(ChatMessage.model_validate(item))
            except Exception:
                continue
        return history

    def _save_history(self) -> None:
        try:
            self._history_file.parent.mkdir(parents=True, exist_ok=True)
            data = [msg.model_dump() for msg in self._history]
            self._history_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            # Не прерываем чат, если не удалось сохранить историю.
            return
