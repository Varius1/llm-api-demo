"""Инварианты ассистента — неизменяемые правила, которые нельзя нарушать."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path

from platformdirs import user_config_dir
from pydantic import BaseModel, Field

APP_NAME = "llm-cli"
INVARIANTS_FILENAME = "invariants.json"

CATEGORY_LABELS: dict[str, str] = {
    "architecture": "Архитектура",
    "technical": "Технические решения",
    "stack": "Ограничения стека",
    "business": "Бизнес-правила",
}

CATEGORY_ALIASES: dict[str, str] = {
    "arch": "architecture",
    "architecture": "architecture",
    "tech": "technical",
    "technical": "technical",
    "stack": "stack",
    "biz": "business",
    "business": "business",
}


class InvariantCategory(str, Enum):
    ARCHITECTURE = "architecture"
    TECHNICAL = "technical"
    STACK = "stack"
    BUSINESS = "business"


class Invariant(BaseModel):
    id: str = Field(
        default_factory=lambda: f"INV-{uuid.uuid4().hex[:6].upper()}"
    )
    category: InvariantCategory
    title: str
    description: str
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )


class InvariantStore(BaseModel):
    invariants: list[Invariant] = Field(default_factory=list)


def _invariants_path() -> Path:
    config_dir = Path(user_config_dir(APP_NAME, appauthor=False, ensure_exists=True))
    return config_dir / INVARIANTS_FILENAME


class InvariantManager:
    """Хранилище и построитель инвариантов ассистента.

    Инварианты хранятся в отдельном файле invariants.json и инжектируются
    в каждый запрос к модели как системный блок с категорическим запретом нарушений.
    """

    def __init__(self) -> None:
        self._path = _invariants_path()
        self._store = self._load()

    def _load(self) -> InvariantStore:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                return InvariantStore.model_validate(data)
            except Exception:
                return InvariantStore()
        return InvariantStore()

    def _save(self) -> None:
        self._path.write_text(
            self._store.model_dump_json(indent=2), encoding="utf-8"
        )

    @property
    def invariants(self) -> list[Invariant]:
        return list(self._store.invariants)

    def add(
        self,
        category: InvariantCategory,
        title: str,
        description: str,
    ) -> Invariant:
        """Добавить новый инвариант и сохранить."""
        inv = Invariant(category=category, title=title, description=description)
        self._store.invariants.append(inv)
        self._save()
        return inv

    def remove(self, inv_id: str) -> bool:
        """Удалить инвариант по ID. Возвращает True если найден и удалён."""
        before = len(self._store.invariants)
        self._store.invariants = [
            i for i in self._store.invariants if i.id != inv_id
        ]
        changed = len(self._store.invariants) < before
        if changed:
            self._save()
        return changed

    def clear(self) -> None:
        """Удалить все инварианты."""
        self._store.invariants.clear()
        self._save()

    def build_block(self) -> str:
        """Построить system-блок инвариантов для инжекции в запрос к LLM.

        Возвращает пустую строку если инвариантов нет.
        Блок содержит явные инструкции: при конфликте запроса с инвариантом
        ассистент обязан отказать и объяснить нарушение.
        """
        if not self._store.invariants:
            return ""

        lines = [
            "[ИНВАРИАНТЫ ПРОЕКТА — НАРУШАТЬ СТРОГО ЗАПРЕЩЕНО]",
            "",
            "Ниже перечислены неизменяемые правила проекта. Ты ОБЯЗАН явно учитывать их в каждом рассуждении.",
            "",
            "ПРАВИЛА ПОВЕДЕНИЯ ПРИ КОНФЛИКТЕ:",
            "  1. Если запрос пользователя нарушает инвариант — ОТКАЖИСЬ его выполнять.",
            "  2. Явно назови нарушенный инвариант: его ID и название.",
            "  3. Объясни конкретно, почему запрос противоречит этому правилу.",
            "  4. Предложи допустимую альтернативу, если она существует.",
            "  5. Не ищи обходных путей и не выполняй запрос «частично» — либо полное соответствие, либо отказ.",
            "",
        ]

        by_category: dict[InvariantCategory, list[Invariant]] = {}
        for inv in self._store.invariants:
            by_category.setdefault(inv.category, []).append(inv)

        for category in InvariantCategory:
            items = by_category.get(category)
            if not items:
                continue
            label = CATEGORY_LABELS[category.value].upper()
            lines.append(f"── {label} ──")
            for inv in items:
                lines.append(f"• [{inv.id}] {inv.title}")
                lines.append(f"  Правило: {inv.description}")
            lines.append("")

        return "\n".join(lines).rstrip()
