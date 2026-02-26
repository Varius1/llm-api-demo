"""LLM-агент — инкапсулирует логику диалога с моделью."""

from __future__ import annotations

import json
import re
from pathlib import Path

from platformdirs import user_config_dir

from .api import OpenRouterClient
from .models import (
    ChatMessage,
    ChatTurnStats,
    CompressionStatus,
    TokenUsage,
    calculate_usage_cost_usd,
)

APP_NAME = "llm-cli"
HISTORY_FILENAME = "history.json"
DEFAULT_KEEP_LAST_N = 10
DEFAULT_SUMMARIZE_EVERY = 10
DEFAULT_MIN_MESSAGES_FOR_SUMMARY = 25
SUMMARY_MAX_CHARS = 520
DEFAULT_SYSTEM_PROMPT = (
    "Ты полезный ассистент для CLI-интерфейса. "
    "Всегда отвечай пользователю только на русском языке."
)


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
        *,
        compression_enabled: bool = True,
        keep_last_n: int = DEFAULT_KEEP_LAST_N,
        summarize_every: int = DEFAULT_SUMMARIZE_EVERY,
        min_messages_for_summary: int = DEFAULT_MIN_MESSAGES_FOR_SUMMARY,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._history_file = _history_path()
        self._raw_history: list[ChatMessage] = []
        self._summary_text = ""
        self._summary_source_messages = 0
        self._compression_enabled = compression_enabled
        self._keep_last_n = max(1, keep_last_n)
        self._summarize_every = max(1, summarize_every)
        self._min_messages_for_summary = max(1, min_messages_for_summary)
        self._load_state()
        self._restored_messages_count = len(self._raw_history)
        self._session_prompt_tokens = 0
        self._session_completion_tokens = 0
        self._session_total_tokens = 0
        self._session_cost_usd = 0.0
        self._last_turn_stats: ChatTurnStats | None = None

        self._ensure_default_system_prompt()
        if system_prompt:
            has_system = any(msg.role == "system" for msg in self._raw_history)
            if not has_system:
                self._raw_history.insert(0, ChatMessage(role="system", content=system_prompt))
                self._save_state()

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
        return list(self._raw_history)

    @property
    def restored_messages_count(self) -> int:
        return self._restored_messages_count

    @property
    def compression_enabled(self) -> bool:
        return self._compression_enabled

    @compression_enabled.setter
    def compression_enabled(self, value: bool) -> None:
        self._compression_enabled = value
        self._save_state()

    @property
    def compression_status(self) -> CompressionStatus:
        _, dialog_messages = _split_system_and_dialog(self._raw_history)
        compressed_count = max(0, len(dialog_messages) - self._keep_last_n)
        return CompressionStatus(
            enabled=self._compression_enabled,
            keep_last_n=self._keep_last_n,
            summarize_every=self._summarize_every,
            min_messages_for_summary=self._min_messages_for_summary,
            summary_chars=len(self._summary_text),
            compressed_messages_count=compressed_count,
        )

    @property
    def last_turn_stats(self) -> ChatTurnStats | None:
        return self._last_turn_stats

    def get_session_totals(self) -> tuple[int, int, int, float]:
        return (
            self._session_prompt_tokens,
            self._session_completion_tokens,
            self._session_total_tokens,
            self._session_cost_usd,
        )

    def reset_session_metrics(self) -> None:
        self._session_prompt_tokens = 0
        self._session_completion_tokens = 0
        self._session_total_tokens = 0
        self._session_cost_usd = 0.0
        self._last_turn_stats = None

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
        self._raw_history.append(user_message)
        previous_summary = self._summary_text
        previous_summary_source = self._summary_source_messages

        request_tokens_estimated = _estimate_text_tokens(user_input)
        messages_for_request, compression_meta = self._build_messages_for_request()
        raw_history_tokens_estimated = _estimate_messages_tokens(self._raw_history)
        sent_history_tokens_estimated = _estimate_messages_tokens(messages_for_request)

        try:
            reply, usage = self._client.send_with_usage(
                messages_for_request,
                self._model,
                self._temperature,
                transforms=transforms,
            )
        except Exception:
            # Не сохраняем в истории неуспешный пользовательский ход.
            self._raw_history.pop()
            self._summary_text = previous_summary
            self._summary_source_messages = previous_summary_source
            raise

        self._raw_history.append(ChatMessage(role="assistant", content=reply))
        self._save_state()

        stats = self._build_turn_stats(
            usage=usage,
            request_tokens_estimated=request_tokens_estimated,
            raw_history_tokens_estimated=raw_history_tokens_estimated,
            sent_history_tokens_estimated=sent_history_tokens_estimated,
            compression_meta=compression_meta,
        )
        self._last_turn_stats = stats
        return reply, stats

    def clear_history(self) -> None:
        """Очистить историю, сохранив системный промпт (если был)."""
        system = [m for m in self._raw_history if m.role == "system"]
        self._raw_history = system
        self._summary_text = ""
        self._summary_source_messages = 0
        self._save_state()
        self._last_turn_stats = None

    def _load_state(self) -> None:
        if not self._history_file.exists():
            return

        try:
            raw_text = self._history_file.read_text(encoding="utf-8")
            if not raw_text.strip():
                return
            payload = json.loads(raw_text)
        except (OSError, json.JSONDecodeError):
            return

        # Обратная совместимость со старым форматом: просто список сообщений.
        if isinstance(payload, list):
            self._raw_history = _validate_messages(payload)
            self._summary_text = ""
            self._summary_source_messages = 0
            return

        if not isinstance(payload, dict):
            return

        self._raw_history = _validate_messages(payload.get("raw_history", []))
        self._summary_text = str(payload.get("summary_text", "") or "")
        self._summary_source_messages = int(payload.get("summary_source_messages", 0) or 0)

        compression = payload.get("compression", {})
        if isinstance(compression, dict):
            self._compression_enabled = bool(
                compression.get("enabled", self._compression_enabled)
            )
            self._keep_last_n = max(
                1,
                int(compression.get("keep_last_n", self._keep_last_n) or self._keep_last_n),
            )
            self._summarize_every = max(
                1,
                int(
                    compression.get("summarize_every", self._summarize_every)
                    or self._summarize_every
                ),
            )
            self._min_messages_for_summary = max(
                1,
                int(
                    compression.get(
                        "min_messages_for_summary", self._min_messages_for_summary
                    )
                    or self._min_messages_for_summary
                ),
            )

    def _save_state(self) -> None:
        try:
            self._history_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "raw_history": [msg.model_dump() for msg in self._raw_history],
                "summary_text": self._summary_text,
                "summary_source_messages": self._summary_source_messages,
                "compression": {
                    "enabled": self._compression_enabled,
                    "keep_last_n": self._keep_last_n,
                    "summarize_every": self._summarize_every,
                    "min_messages_for_summary": self._min_messages_for_summary,
                },
            }
            self._history_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            # Не прерываем чат, если не удалось сохранить историю.
            return

    def _build_messages_for_request(self) -> tuple[list[ChatMessage], dict[str, int | bool]]:
        if not self._compression_enabled:
            return list(self._raw_history), {
                "used_summary": False,
                "summary_chars": len(self._summary_text),
                "compressed_messages_count": 0,
            }

        system_messages, dialog_messages = _split_system_and_dialog(self._raw_history)
        if len(dialog_messages) < self._min_messages_for_summary:
            return list(self._raw_history), {
                "used_summary": False,
                "summary_chars": len(self._summary_text),
                "compressed_messages_count": 0,
            }
        if len(dialog_messages) <= self._keep_last_n:
            if self._summary_text:
                self._summary_text = ""
                self._summary_source_messages = 0
                self._save_state()
            return list(self._raw_history), {
                "used_summary": False,
                "summary_chars": 0,
                "compressed_messages_count": 0,
            }

        archive_messages = dialog_messages[:-self._keep_last_n]
        tail_messages = dialog_messages[-self._keep_last_n :]
        self._refresh_summary_if_needed(archive_messages)

        messages_for_request = list(system_messages)
        used_summary = bool(self._summary_text.strip())
        if used_summary:
            messages_for_request.append(
                ChatMessage(
                    role="system",
                    content=_build_summary_system_message(self._summary_text),
                )
            )
        messages_for_request.extend(tail_messages)

        # Защита: если сжатый контекст не меньше исходного, не используем summary.
        raw_estimated = _estimate_messages_tokens(self._raw_history)
        compressed_estimated = _estimate_messages_tokens(messages_for_request)
        if compressed_estimated >= raw_estimated:
            return list(self._raw_history), {
                "used_summary": False,
                "summary_chars": len(self._summary_text),
                "compressed_messages_count": 0,
            }

        return messages_for_request, {
            "used_summary": used_summary,
            "summary_chars": len(self._summary_text),
            "compressed_messages_count": len(archive_messages),
        }

    def _refresh_summary_if_needed(self, archive_messages: list[ChatMessage]) -> None:
        archive_count = len(archive_messages)
        if archive_count == 0:
            if self._summary_text:
                self._summary_text = ""
                self._summary_source_messages = 0
                self._save_state()
            return

        should_refresh = (
            not self._summary_text
            or (archive_count - self._summary_source_messages) >= self._summarize_every
        )
        if not should_refresh:
            return

        self._summary_text = _build_heuristic_summary(archive_messages)
        self._summary_source_messages = archive_count
        self._save_state()

    def _build_turn_stats(
        self,
        usage: TokenUsage | None,
        request_tokens_estimated: int,
        raw_history_tokens_estimated: int,
        sent_history_tokens_estimated: int,
        compression_meta: dict[str, int | bool],
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
            history_tokens_estimated=sent_history_tokens_estimated,
            raw_history_tokens_estimated=raw_history_tokens_estimated,
            sent_history_tokens_estimated=sent_history_tokens_estimated,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            turn_cost_usd=turn_cost,
            compression_enabled=self._compression_enabled,
            used_summary=bool(compression_meta.get("used_summary", False)),
            summary_chars=int(compression_meta.get("summary_chars", 0)),
            compressed_messages_count=int(compression_meta.get("compressed_messages_count", 0)),
            session_prompt_tokens=self._session_prompt_tokens,
            session_completion_tokens=self._session_completion_tokens,
            session_total_tokens=self._session_total_tokens,
            session_cost_usd=self._session_cost_usd,
        )

    def _ensure_default_system_prompt(self) -> None:
        has_system = any(msg.role == "system" for msg in self._raw_history)
        if has_system:
            return
        self._raw_history.insert(0, ChatMessage(role="system", content=DEFAULT_SYSTEM_PROMPT))
        self._save_state()


def _estimate_text_tokens(text: str) -> int:
    # Грубая эвристика для CLI-демонстрации: ~1 токен на 4 символа.
    return max(1, (len(text) + 3) // 4)


def _estimate_messages_tokens(messages: list[ChatMessage]) -> int:
    # Небольшой накладной расход на role/структуру + токены контента.
    return sum(_estimate_text_tokens(msg.content) + 4 for msg in messages)


def _validate_messages(items: list[object]) -> list[ChatMessage]:
    history: list[ChatMessage] = []
    for item in items:
        try:
            history.append(ChatMessage.model_validate(item))
        except Exception:
            continue
    return history


def _split_system_and_dialog(
    messages: list[ChatMessage],
) -> tuple[list[ChatMessage], list[ChatMessage]]:
    system_messages = [m for m in messages if m.role == "system"]
    dialog_messages = [m for m in messages if m.role != "system"]
    return system_messages, dialog_messages


def _build_summary_system_message(summary_text: str) -> str:
    return (
        "Сводка более ранней части диалога (используется вместо полной старой истории):\n"
        f"{summary_text}\n\n"
        "Используй только факты из сводки и последних сообщений. "
        "Если данных недостаточно — явно так и скажи, не выдумывай. "
        "Отвечай на русском, кратко и по делу."
    )


def _build_heuristic_summary(messages: list[ChatMessage]) -> str:
    project: str | None = None
    client: str | None = None
    deadline: str | None = None
    budget: str | None = None
    team: str | None = None
    stack: str | None = None

    priorities: list[str] = []
    constraints: list[str] = []
    risks: list[str] = []
    requirements: list[str] = []
    kpis: list[str] = []

    for message in messages:
        cleaned = _normalize_text(message.content)
        if not cleaned:
            continue
        if message.role != "user":
            continue
        lowered = cleaned.lower()
        snippet = _shrink_snippet(cleaned, max_len=110)

        if project is None and ("проект называется" in lowered or lowered.startswith("проект:")):
            project = _extract_value(cleaned)
            continue
        if client is None and ("клиент" in lowered and ":" in cleaned):
            client = _extract_value(cleaned)
            continue
        if deadline is None and "дедлайн" in lowered:
            deadline = _extract_value(cleaned)
            continue
        if budget is None and "бюджет" in lowered:
            budget = _extract_value(cleaned)
            continue
        if team is None and "команда" in lowered:
            team = _extract_value(cleaned)
            continue
        if stack is None and "стек" in lowered:
            stack = _extract_value(cleaned)
            continue

        if "приоритет" in lowered:
            priorities.append(snippet)
            continue
        if "риск" in lowered:
            risks.append(snippet)
            continue
        if "ограничение" in lowered:
            constraints.append(snippet)
            continue
        if "требование" in lowered or "правило" in lowered:
            requirements.append(snippet)
            continue
        if "kpi" in lowered or "sla" in lowered:
            kpis.append(snippet)
            continue

    lines = [
        "Сжатая память диалога (только подтверждённые факты):",
        "",
        f"- Проект: {project or 'не указано'}",
        f"- Клиент: {client or 'не указано'}",
        f"- Дедлайн: {deadline or 'не указано'}",
        f"- Бюджет: {budget or 'не указано'}",
        f"- Команда: {team or 'не указано'}",
        f"- Стек: {stack or 'не указано'}",
        "",
        "Приоритеты:",
    ]
    lines.extend(f"- {item}" for item in _deduplicate_keep_last(priorities)[-3:] or ["(нет данных)"])
    lines.append("")
    lines.append("Ограничения:")
    lines.extend(
        f"- {item}" for item in _deduplicate_keep_last(constraints)[-3:] or ["(нет данных)"]
    )
    lines.append("")
    lines.append("Риски:")
    lines.extend(f"- {item}" for item in _deduplicate_keep_last(risks)[-3:] or ["(нет данных)"])
    lines.append("")
    lines.append("Требования/KPI:")
    req_kpi = _deduplicate_keep_last(requirements + kpis)[-4:]
    lines.extend(f"- {item}" for item in req_kpi or ["(нет данных)"])

    summary = "\n".join(lines).strip()
    if len(summary) <= SUMMARY_MAX_CHARS:
        return summary

    # Ультра-компактная версия для строгого лимита.
    compact_lines = [
        f"Проект={project or 'не указано'}; Клиент={client or 'не указано'}; "
        f"Дедлайн={deadline or 'не указано'}; Бюджет={budget or 'не указано'};",
        f"Команда={team or 'не указано'}; Стек={stack or 'не указано'};",
        "Ограничения: " + "; ".join(_deduplicate_keep_last(constraints)[-2:] or ["нет данных"]),
        "Риски: " + "; ".join(_deduplicate_keep_last(risks)[-2:] or ["нет данных"]),
        "Требования: " + "; ".join(req_kpi[-2:] or ["нет данных"]),
    ]
    compact = "\n".join(compact_lines).strip()
    return compact[:SUMMARY_MAX_CHARS]


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _shrink_snippet(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _deduplicate_keep_last(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out_reversed: list[str] = []
    for item in reversed(items):
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out_reversed.append(item)
    return list(reversed(out_reversed))


def _extract_value(text: str) -> str:
    if ":" in text:
        value = text.split(":", 1)[1].strip()
        if value:
            return _shrink_snippet(value, max_len=90)

    m = re.search(r"(?i)проект называется\s+(.+)", text)
    if m:
        return _shrink_snippet(m.group(1).strip(), max_len=90)

    return _shrink_snippet(text, max_len=90)
