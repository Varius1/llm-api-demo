"""Стратегии управления контекстом для LLM-агента."""

from __future__ import annotations

import re

from .models import ChatMessage, StrategyType

__all__ = [
    "StrategyType",
    "build_sliding_window",
    "extract_facts",
    "build_facts_block",
    "STRATEGY_LABELS",
]

STRATEGY_LABELS: dict[StrategyType, str] = {
    StrategyType.SLIDING_WINDOW: "Sliding Window (последние N сообщений)",
    StrategyType.STICKY_FACTS: "Sticky Facts (KV-память + последние N)",
    StrategyType.SUMMARY: "Summary (эвристическое сжатие)",
    StrategyType.BRANCHING: "Branching (ветки диалога)",
}

# ─────────────────────────────────────────────────────────────────────────────
# Стратегия 1: Sliding Window
# ─────────────────────────────────────────────────────────────────────────────

def build_sliding_window(
    messages: list[ChatMessage],
    keep_n: int,
) -> list[ChatMessage]:
    """Вернуть системные сообщения + последние keep_n диалоговых сообщений."""
    system_msgs = [m for m in messages if m.role == "system"]
    dialog_msgs = [m for m in messages if m.role != "system"]
    tail = dialog_msgs[-keep_n:] if keep_n > 0 else []
    return system_msgs + tail


# ─────────────────────────────────────────────────────────────────────────────
# Стратегия 2: Sticky Facts
# ─────────────────────────────────────────────────────────────────────────────

# Поля фактов и ключевые слова для их детектирования.
_FACT_RULES: list[tuple[str, list[str]]] = [
    ("проект",       ["проект называется", "проект:"]),
    ("клиент",       ["клиент:"]),
    ("дедлайн",      ["дедлайн"]),
    ("бюджет",       ["бюджет"]),
    ("команда",      ["команда"]),
    ("стек",         ["стек"]),
    ("приоритет",    ["приоритет"]),
    ("риски",        ["риск"]),
    ("ограничения",  ["ограничение"]),
    ("требования",   ["требование", "правило"]),
    ("kpi_sla",      ["kpi", "sla"]),
    ("безопасность", ["безопасность", "доступ", "роль", "admin"]),
    ("интеграции",   ["интеграция", "интеграции"]),
    ("деплой",       ["деплой", "инфраструктур"]),
]


def extract_facts(text: str, existing: dict[str, str]) -> dict[str, str]:
    """Обновить facts из текста сообщения пользователя.

    Для полей-одиночек (проект, клиент, дедлайн, бюджет, команда, стек)
    значение перезаписывается. Для списочных полей (риски, ограничения и т.д.)
    новое значение добавляется через «; » к старому (дедупликация).
    """
    updated = dict(existing)
    lowered = text.lower()
    cleaned = " ".join(text.split())

    for key, triggers in _FACT_RULES:
        matched = any(t in lowered for t in triggers)
        if not matched:
            continue

        value = _extract_value(cleaned)
        if not value:
            continue

        # Одиночные поля — просто перезаписываем.
        if key in ("проект", "клиент", "дедлайн", "бюджет", "команда", "стек"):
            updated[key] = value
        else:
            # Списочные поля — аккумулируем без дубликатов.
            existing_val = updated.get(key, "")
            parts = [p.strip() for p in existing_val.split(";") if p.strip()]
            if value not in parts:
                parts.append(value)
            updated[key] = "; ".join(parts[-5:])  # не более 5 значений на поле

    return updated


def build_facts_block(facts: dict[str, str]) -> str:
    """Форматировать facts как текст для system-сообщения."""
    if not facts:
        return ""
    lines = ["Известные факты из диалога (использовать как опорный контекст):"]
    for key, value in facts.items():
        lines.append(f"- {key}: {value}")
    lines.append(
        "\nЕсли в текущем сообщении пользователя встречается противоречие с фактами — "
        "используй данные из текущего сообщения как более актуальные."
    )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────────────────────────────────────

def _extract_value(text: str) -> str:
    """Извлечь значение из строки вида 'ключ: значение' или 'проект называется X'."""
    m = re.search(r"(?i)проект называется\s+(.+)", text)
    if m:
        return _shrink(m.group(1).strip(), 100)

    if ":" in text:
        value = text.split(":", 1)[1].strip()
        if value:
            return _shrink(value, 100)

    return _shrink(text, 100)


def _shrink(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."
