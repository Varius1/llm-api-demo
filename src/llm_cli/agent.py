"""LLM-агент — инкапсулирует логику диалога с моделью."""

from __future__ import annotations

import json
from pathlib import Path

from platformdirs import user_config_dir

from .api import OpenRouterClient
from .models import ChatMessage, ChatTurnStats, TokenUsage, calculate_usage_cost_usd

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
        self._session_prompt_tokens = 0
        self._session_completion_tokens = 0
        self._session_total_tokens = 0
        self._session_cost_usd = 0.0
        self._last_turn_stats: ChatTurnStats | None = None

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

    @property
    def last_turn_stats(self) -> ChatTurnStats | None:
        return self._last_turn_stats

    def run(self, user_input: str, transforms: list[str] | None = None) -> str:
        reply, _ = self.run_with_stats(user_input, transforms=transforms)
        return reply

    def run_with_stats(
        self, user_input: str, transforms: list[str] | None = None
    ) -> tuple[str, ChatTurnStats]:
        """Отправить сообщение пользователя в LLM и вернуть ответ.

        Автоматически добавляет сообщение и ответ в историю,
        так что следующий вызов ``run()`` будет содержать весь контекст.
        """
        user_message = ChatMessage(role="user", content=user_input)
        self._history.append(user_message)

        request_tokens_estimated = _estimate_text_tokens(user_input)
        history_tokens_estimated = _estimate_messages_tokens(self._history)

        try:
            reply, usage = self._client.send_with_usage(
                self._history,
                self._model,
                self._temperature,
                transforms=transforms,
            )
        except Exception:
            # Не сохраняем в истории неуспешный пользовательский ход.
            self._history.pop()
            raise

        self._history.append(ChatMessage(role="assistant", content=reply))
        self._save_history()

        stats = self._build_turn_stats(
            usage=usage,
            request_tokens_estimated=request_tokens_estimated,
            history_tokens_estimated=history_tokens_estimated,
        )
        self._last_turn_stats = stats
        return reply, stats

    def clear_history(self) -> None:
        """Очистить историю, сохранив системный промпт (если был)."""
        system = [m for m in self._history if m.role == "system"]
        self._history = system
        self._save_history()
        self._last_turn_stats = None

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

    def _build_turn_stats(
        self,
        usage: TokenUsage | None,
        request_tokens_estimated: int,
        history_tokens_estimated: int,
    ) -> ChatTurnStats:
        prompt_tokens = usage.prompt_tokens if usage else None
        completion_tokens = usage.completion_tokens if usage else None
        total_tokens = usage.total_tokens if usage else None
        turn_cost = calculate_usage_cost_usd(usage, self._model)

        if usage is not None:
            self._session_prompt_tokens += usage.prompt_tokens
            self._session_completion_tokens += usage.completion_tokens
            self._session_total_tokens += usage.total_tokens
        if turn_cost is not None:
            self._session_cost_usd += turn_cost

        return ChatTurnStats(
            request_tokens_estimated=request_tokens_estimated,
            history_tokens_estimated=history_tokens_estimated,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            turn_cost_usd=turn_cost,
            session_prompt_tokens=self._session_prompt_tokens,
            session_completion_tokens=self._session_completion_tokens,
            session_total_tokens=self._session_total_tokens,
            session_cost_usd=self._session_cost_usd,
        )


def _estimate_text_tokens(text: str) -> int:
    # Грубая эвристика для CLI-демонстрации: ~1 токен на 4 символа.
    return max(1, (len(text) + 3) // 4)


def _estimate_messages_tokens(messages: list[ChatMessage]) -> int:
    # Небольшой накладной расход на role/структуру + токены контента.
    return sum(_estimate_text_tokens(msg.content) + 4 for msg in messages)
