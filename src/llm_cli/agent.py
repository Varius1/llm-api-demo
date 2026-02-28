"""LLM-агент — инкапсулирует логику диалога с моделью."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from platformdirs import user_config_dir

from .api import OpenRouterClient
from .models import (
    BranchInfo,
    ChatMessage,
    ChatTurnStats,
    CompressionStatus,
    StrategyType,
    TokenUsage,
    calculate_usage_cost_usd,
)
from .strategy import build_facts_block, build_sliding_window, extract_facts

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

    Поддерживает 4 стратегии управления контекстом:
    - SLIDING_WINDOW: последние N сообщений, остальное отбрасывается.
    - STICKY_FACTS:   KV-память фактов + последние N сообщений.
    - SUMMARY:        эвристическое сжатие старой части диалога (исходная стратегия).
    - BRANCHING:      ветки диалога с независимой историей.
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

        # Общая история (полная, не обрезанная).
        self._raw_history: list[ChatMessage] = []

        # Summary-стратегия.
        self._summary_text = ""
        self._summary_source_messages = 0

        # Sticky Facts — словарь ключ→значение.
        self._facts: dict[str, str] = {}

        # Branching — хранилище веток.
        self._branches: dict[str, dict[str, object]] = {}
        self._current_branch: str | None = None

        # Настройки стратегий.
        self._strategy: StrategyType = StrategyType.SUMMARY
        self._compression_enabled = compression_enabled
        self._keep_last_n = max(1, keep_last_n)
        self._summarize_every = max(1, summarize_every)
        self._min_messages_for_summary = max(1, min_messages_for_summary)

        self._load_state()
        self._restored_messages_count = len(self._raw_history)

        # Метрики сессии.
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

    # ─────────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────────

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
        """Обратная совместимость: True когда стратегия — SUMMARY."""
        return self._strategy == StrategyType.SUMMARY

    @compression_enabled.setter
    def compression_enabled(self, value: bool) -> None:
        self._strategy = StrategyType.SUMMARY if value else StrategyType.SLIDING_WINDOW
        self._save_state()

    @property
    def strategy(self) -> StrategyType:
        return self._strategy

    @strategy.setter
    def strategy(self, value: StrategyType) -> None:
        self._strategy = value
        self._save_state()

    @property
    def facts(self) -> dict[str, str]:
        return dict(self._facts)

    @property
    def compression_status(self) -> CompressionStatus:
        _, dialog_messages = _split_system_and_dialog(self._raw_history)
        compressed_count = max(0, len(dialog_messages) - self._keep_last_n)
        return CompressionStatus(
            enabled=self._strategy == StrategyType.SUMMARY,
            keep_last_n=self._keep_last_n,
            summarize_every=self._summarize_every,
            min_messages_for_summary=self._min_messages_for_summary,
            summary_chars=len(self._summary_text),
            compressed_messages_count=compressed_count,
            strategy=self._strategy.value,
        )

    @property
    def last_turn_stats(self) -> ChatTurnStats | None:
        return self._last_turn_stats

    # ─────────────────────────────────────────────────────────────────────────
    # Session metrics
    # ─────────────────────────────────────────────────────────────────────────

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

    # ─────────────────────────────────────────────────────────────────────────
    # Public chat API
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, user_input: str, transforms: list[str] | None = None) -> str:
        reply, _ = self.run_with_stats(user_input, transforms=transforms)
        return reply

    def run_with_stats(
        self, user_input: str, transforms: list[str] | None = None
    ) -> tuple[str, ChatTurnStats]:
        """Отправить сообщение пользователя в LLM и вернуть ответ со статистикой."""
        user_message = ChatMessage(role="user", content=user_input)
        self._raw_history.append(user_message)
        previous_summary = self._summary_text
        previous_summary_source = self._summary_source_messages

        # Обновить KV-факты до отправки запроса.
        if self._strategy == StrategyType.STICKY_FACTS:
            self._facts = extract_facts(user_input, self._facts)

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
        self._facts = {}
        self._current_branch = None
        self._save_state()
        self._last_turn_stats = None

    # ─────────────────────────────────────────────────────────────────────────
    # Branching API
    # ─────────────────────────────────────────────────────────────────────────

    def branch_save(self, name: str) -> None:
        """Сохранить снимок текущей истории как ветку с данным именем."""
        self._branches[name] = {
            "history": [msg.model_dump() for msg in self._raw_history],
            "facts": dict(self._facts),
            "summary_text": self._summary_text,
            "summary_source_messages": self._summary_source_messages,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        self._current_branch = name
        self._save_state()

    def branch_switch(self, name: str) -> None:
        """Переключиться на сохранённую ветку."""
        if name not in self._branches:
            raise ValueError(f"Ветка «{name}» не найдена. Доступные: {list(self._branches)}")
        snap = self._branches[name]
        self._raw_history = _validate_messages(snap.get("history", []))  # type: ignore[arg-type]
        self._facts = dict(snap.get("facts", {}))  # type: ignore[arg-type]
        self._summary_text = str(snap.get("summary_text", "") or "")
        self._summary_source_messages = int(snap.get("summary_source_messages", 0) or 0)
        self._current_branch = name
        self._save_state()

    def branch_list(self) -> list[BranchInfo]:
        """Вернуть список сохранённых веток."""
        result: list[BranchInfo] = []
        for name, snap in self._branches.items():
            msgs = snap.get("history", [])
            result.append(
                BranchInfo(
                    name=name,
                    created_at=str(snap.get("created_at", "")),
                    messages_count=len(msgs) if isinstance(msgs, list) else 0,  # type: ignore[arg-type]
                )
            )
        return result

    @property
    def current_branch(self) -> str | None:
        return self._current_branch

    # ─────────────────────────────────────────────────────────────────────────
    # Internal: build messages for request
    # ─────────────────────────────────────────────────────────────────────────

    def _build_messages_for_request(self) -> tuple[list[ChatMessage], dict[str, int | bool]]:
        """Диспетчер по стратегии: вернуть список сообщений для отправки в API."""
        if self._strategy == StrategyType.SLIDING_WINDOW:
            return self._build_sliding_window_request()
        if self._strategy == StrategyType.STICKY_FACTS:
            return self._build_sticky_facts_request()
        if self._strategy == StrategyType.BRANCHING:
            return self._build_branching_request()
        # По умолчанию — summary (исходная логика).
        return self._build_summary_request()

    def _build_sliding_window_request(self) -> tuple[list[ChatMessage], dict[str, int | bool]]:
        msgs = build_sliding_window(self._raw_history, self._keep_last_n)
        dropped = max(0, len([m for m in self._raw_history if m.role != "system"]) - self._keep_last_n)
        return msgs, {
            "used_summary": False,
            "summary_chars": 0,
            "compressed_messages_count": dropped,
        }

    def _build_sticky_facts_request(self) -> tuple[list[ChatMessage], dict[str, int | bool]]:
        system_msgs, dialog_msgs = _split_system_and_dialog(self._raw_history)
        tail = dialog_msgs[-self._keep_last_n:] if self._keep_last_n > 0 else []

        messages: list[ChatMessage] = list(system_msgs)
        facts_block = build_facts_block(self._facts)
        if facts_block:
            messages.append(ChatMessage(role="system", content=facts_block))
        messages.extend(tail)

        dropped = max(0, len(dialog_msgs) - self._keep_last_n)
        return messages, {
            "used_summary": bool(facts_block),
            "summary_chars": len(facts_block),
            "compressed_messages_count": dropped,
        }

    def _build_branching_request(self) -> tuple[list[ChatMessage], dict[str, int | bool]]:
        # В режиме branching используем sliding window по умолчанию.
        msgs = build_sliding_window(self._raw_history, self._keep_last_n)
        dropped = max(0, len([m for m in self._raw_history if m.role != "system"]) - self._keep_last_n)
        return msgs, {
            "used_summary": False,
            "summary_chars": 0,
            "compressed_messages_count": dropped,
        }

    def _build_summary_request(self) -> tuple[list[ChatMessage], dict[str, int | bool]]:
        """Исходная логика summary-сжатия."""
        if not self._compression_enabled and self._strategy != StrategyType.SUMMARY:
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

        archive_messages = dialog_messages[: -self._keep_last_n]
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

    # ─────────────────────────────────────────────────────────────────────────
    # Internal: summary helpers
    # ─────────────────────────────────────────────────────────────────────────

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

    # ─────────────────────────────────────────────────────────────────────────
    # Internal: stats
    # ─────────────────────────────────────────────────────────────────────────

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
            compression_enabled=self._strategy == StrategyType.SUMMARY,
            used_summary=bool(compression_meta.get("used_summary", False)),
            summary_chars=int(compression_meta.get("summary_chars", 0)),
            compressed_messages_count=int(compression_meta.get("compressed_messages_count", 0)),
            strategy=self._strategy.value,
            session_prompt_tokens=self._session_prompt_tokens,
            session_completion_tokens=self._session_completion_tokens,
            session_total_tokens=self._session_total_tokens,
            session_cost_usd=self._session_cost_usd,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Internal: persistence
    # ─────────────────────────────────────────────────────────────────────────

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
            return

        if not isinstance(payload, dict):
            return

        self._raw_history = _validate_messages(payload.get("raw_history", []))
        self._summary_text = str(payload.get("summary_text", "") or "")
        self._summary_source_messages = int(payload.get("summary_source_messages", 0) or 0)

        raw_facts = payload.get("facts", {})
        if isinstance(raw_facts, dict):
            self._facts = {str(k): str(v) for k, v in raw_facts.items()}

        raw_branches = payload.get("branches", {})
        if isinstance(raw_branches, dict):
            self._branches = raw_branches  # type: ignore[assignment]

        self._current_branch = payload.get("current_branch") or None

        raw_strategy = payload.get("strategy")
        if raw_strategy:
            try:
                self._strategy = StrategyType(raw_strategy)
            except ValueError:
                pass

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
                    compression.get("min_messages_for_summary", self._min_messages_for_summary)
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
                "facts": self._facts,
                "branches": self._branches,
                "current_branch": self._current_branch,
                "strategy": self._strategy.value,
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
            return

    def _ensure_default_system_prompt(self) -> None:
        has_system = any(msg.role == "system" for msg in self._raw_history)
        if has_system:
            return
        self._raw_history.insert(0, ChatMessage(role="system", content=DEFAULT_SYSTEM_PROMPT))
        self._save_state()


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

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
