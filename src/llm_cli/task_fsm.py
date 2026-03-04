"""Детерминированный конечный автомат задачи (Task FSM).

Жизненный цикл:
    planning → execution → validation → done
    любой этап ↔ paused (пауза/возобновление)

Переходы управляются только кодом — LLM не принимает решений о смене этапа.
Каждый активный этап инжектируется как отдельный system-промпт в каждый запрос.
При паузе промпт убирается: LLM отвечает свободно, затем при resume возвращается
в нужный этап без повторного объяснения контекста.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class TaskStage(str, Enum):
    PLANNING = "planning"
    EXECUTION = "execution"
    VALIDATION = "validation"
    DONE = "done"
    PAUSED = "paused"


STAGE_LABELS: dict[TaskStage, str] = {
    TaskStage.PLANNING: "Планирование",
    TaskStage.EXECUTION: "Реализация",
    TaskStage.VALIDATION: "Валидация",
    TaskStage.DONE: "Завершено",
    TaskStage.PAUSED: "Пауза",
}

# Линейная цепочка активных этапов (без PAUSED и DONE)
STAGE_CHAIN: list[TaskStage] = [
    TaskStage.PLANNING,
    TaskStage.EXECUTION,
    TaskStage.VALIDATION,
    TaskStage.DONE,
]

ALLOWED_TRANSITIONS: set[tuple[TaskStage, TaskStage]] = {
    (TaskStage.PLANNING, TaskStage.EXECUTION),
    (TaskStage.EXECUTION, TaskStage.VALIDATION),
    (TaskStage.VALIDATION, TaskStage.DONE),
    (TaskStage.PLANNING, TaskStage.PAUSED),
    (TaskStage.EXECUTION, TaskStage.PAUSED),
    (TaskStage.VALIDATION, TaskStage.PAUSED),
    # PAUSED → любой этап из STAGE_CHAIN — обрабатывается через resume()
}

# System-промпты, инжектируемые агентом при каждом запросе к LLM.
# PAUSED и DONE намеренно отсутствуют — при этих статусах промпт не инжектируется.
STAGE_SYSTEM_PROMPTS: dict[TaskStage, str] = {
    TaskStage.PLANNING: (
        "[FSM: ЭТАП ПЛАНИРОВАНИЯ]\n"
        "Задача сейчас находится на этапе планирования.\n"
        "Твоя роль — помочь пользователю:\n"
        "- чётко сформулировать требования и цели задачи,\n"
        "- разбить задачу на конкретные шаги,\n"
        "- выявить риски и неизвестные.\n"
        "Не начинай реализацию. Итог этапа — задокументированный план."
    ),
    TaskStage.EXECUTION: (
        "[FSM: ЭТАП РЕАЛИЗАЦИИ]\n"
        "Задача сейчас на этапе реализации.\n"
        "Твоя роль — помогать выполнять задачу строго по ранее составленному плану:\n"
        "- давай конкретные решения, код, тексты или действия,\n"
        "- не перепланируй и не меняй требования на ходу,\n"
        "- если план нуждается в коррекции — сообщи пользователю.\n"
        "Итог этапа — конкретный результат (артефакт)."
    ),
    TaskStage.VALIDATION: (
        "[FSM: ЭТАП ВАЛИДАЦИИ]\n"
        "Задача сейчас на этапе валидации.\n"
        "Твоя роль — критически проверить результаты реализации:\n"
        "- проверь артефакты на соответствие требованиям из плана,\n"
        "- выяви проблемы, пропуски и пути улучшения,\n"
        "- дай чёткое заключение: принять / доработать / отклонить.\n"
        "Итог этапа — вердикт с обоснованием."
    ),
}


class StageTransition(BaseModel):
    """Запись о переходе между этапами."""

    from_stage: str
    to_stage: str
    timestamp: str
    note: str = ""


class TaskFSM(BaseModel):
    """Состояние конечного автомата задачи."""

    task_name: str = ""
    stage: TaskStage = TaskStage.PLANNING
    current_step: str = ""
    expected_action: str = ""
    artifacts: dict[str, str] = Field(default_factory=dict)
    transitions: list[StageTransition] = Field(default_factory=list)
    paused_at: str = ""
    created_at: str = ""

    # ── Переходы ──────────────────────────────────────────────────────────────

    def _record_transition(self, to_stage: TaskStage, note: str = "") -> None:
        self.transitions.append(
            StageTransition(
                from_stage=self.stage.value,
                to_stage=to_stage.value,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
                note=note,
            )
        )
        self.stage = to_stage

    def can_advance(self) -> bool:
        """Можно ли перейти к следующему этапу линейной цепочки."""
        if self.stage == TaskStage.PAUSED or self.stage == TaskStage.DONE:
            return False
        idx = STAGE_CHAIN.index(self.stage) if self.stage in STAGE_CHAIN else -1
        return 0 <= idx < len(STAGE_CHAIN) - 1

    def next_stage(self) -> TaskStage | None:
        """Следующий этап в цепочке (None если нет следующего)."""
        if self.stage not in STAGE_CHAIN:
            return None
        idx = STAGE_CHAIN.index(self.stage)
        if idx >= len(STAGE_CHAIN) - 1:
            return None
        return STAGE_CHAIN[idx + 1]

    def advance(self, note: str = "") -> None:
        """Перейти к следующему этапу линейной цепочки."""
        nxt = self.next_stage()
        if nxt is None:
            raise ValueError(
                f"Нет следующего этапа после «{STAGE_LABELS[self.stage]}»."
            )
        self._record_transition(nxt, note)
        self.current_step = ""
        self.expected_action = ""

    def pause(self) -> None:
        """Поставить FSM на паузу, сохранив текущий этап."""
        if self.stage == TaskStage.PAUSED:
            raise ValueError("FSM уже на паузе.")
        if self.stage == TaskStage.DONE:
            raise ValueError("Задача завершена, пауза невозможна.")
        self.paused_at = self.stage.value
        self._record_transition(TaskStage.PAUSED)

    def resume(self) -> None:
        """Возобновить FSM с этапа, на котором была поставлена пауза."""
        if self.stage != TaskStage.PAUSED:
            raise ValueError("FSM не на паузе.")
        if not self.paused_at:
            raise ValueError("Нет сохранённого этапа для возобновления.")
        restore = TaskStage(self.paused_at)
        self._record_transition(restore)
        self.paused_at = ""

    def add_artifact(self, key: str, text: str) -> None:
        """Сохранить артефакт текущего этапа."""
        self.artifacts[key.strip()] = text.strip()

    def set_step(self, step: str, expected_action: str = "") -> None:
        """Установить текущий шаг и ожидаемое действие внутри этапа."""
        self.current_step = step.strip()
        self.expected_action = expected_action.strip()

    # ── Инъекция в запрос ─────────────────────────────────────────────────────

    def build_stage_prompt(self) -> str:
        """Вернуть system-промпт для текущего этапа (пустая строка если PAUSED/DONE)."""
        return STAGE_SYSTEM_PROMPTS.get(self.stage, "")

    def build_status_block(self) -> str:
        """Краткий блок FSM-статуса для инъекции в контекст памяти."""
        label = STAGE_LABELS.get(self.stage, self.stage.value)
        lines = [f"[FSM] Задача: «{self.task_name}» | Этап: {label}"]
        if self.stage == TaskStage.PAUSED and self.paused_at:
            paused_label = STAGE_LABELS.get(
                TaskStage(self.paused_at), self.paused_at
            )
            lines.append(f"  Пауза на: {paused_label}")
        if self.current_step:
            lines.append(f"  Шаг: {self.current_step}")
        if self.expected_action:
            lines.append(f"  Ожидается: {self.expected_action}")
        if self.artifacts:
            keys = ", ".join(self.artifacts.keys())
            lines.append(f"  Артефакты: {keys}")
        return "\n".join(lines)
