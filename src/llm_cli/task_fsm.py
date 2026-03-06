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

# Причины блокировки конкретных недопустимых переходов.
# Ключ: (from_stage, to_stage) — строковое сообщение + подсказка что делать.
# Включает как полностью запрещённые переходы, так и переходы запрещённые для goto().
TRANSITION_BLOCK_REASONS: dict[tuple[TaskStage, TaskStage], tuple[str, str]] = {
    # Прямые переходы вперёд через goto — разрешены только через advance (next)
    (TaskStage.PLANNING, TaskStage.EXECUTION): (
        "Нельзя перейти к реализации напрямую — план должен быть явно завершён.",
        "Используйте /task fsm next чтобы подтвердить завершение планирования.",
    ),
    (TaskStage.PLANNING, TaskStage.VALIDATION): (
        "Нельзя перейти к валидации без завершённой реализации.",
        "Следуйте цепочке: planning → execution → validation → done",
    ),
    (TaskStage.PLANNING, TaskStage.DONE): (
        "Нельзя завершить задачу, не пройдя реализацию и валидацию.",
        "Следуйте цепочке: planning → execution → validation → done",
    ),
    (TaskStage.EXECUTION, TaskStage.VALIDATION): (
        "Нельзя перейти к валидации напрямую — реализация должна быть явно завершена.",
        "Используйте /task fsm next чтобы подтвердить завершение реализации.",
    ),
    (TaskStage.EXECUTION, TaskStage.DONE): (
        "Нельзя завершить задачу без валидации.",
        "Следуйте цепочке: execution → validation → done",
    ),
    (TaskStage.EXECUTION, TaskStage.PLANNING): (
        "Нельзя вернуться назад по цепочке этапов.",
        "Переходы возможны только вперёд. Если план нужно скорректировать — сообщите пользователю.",
    ),
    (TaskStage.VALIDATION, TaskStage.DONE): (
        "Нельзя завершить задачу напрямую — валидация должна быть явно завершена.",
        "Используйте /task fsm next чтобы подтвердить вердикт валидации.",
    ),
    (TaskStage.VALIDATION, TaskStage.PLANNING): (
        "Нельзя вернуться к планированию с этапа валидации.",
        "Переходы возможны только вперёд. Завершите валидацию: /task fsm next",
    ),
    (TaskStage.VALIDATION, TaskStage.EXECUTION): (
        "Нельзя вернуться к реализации с этапа валидации.",
        "Если требуется доработка — зафиксируйте это в артефакте и завершите: /task fsm next",
    ),
    (TaskStage.DONE, TaskStage.PLANNING): (
        "Задача уже завершена. Нельзя вернуться к планированию.",
        "Запустите новую задачу: /task fsm start <имя>",
    ),
    (TaskStage.DONE, TaskStage.EXECUTION): (
        "Задача уже завершена. Нельзя вернуться к реализации.",
        "Запустите новую задачу: /task fsm start <имя>",
    ),
    (TaskStage.DONE, TaskStage.VALIDATION): (
        "Задача уже завершена. Нельзя вернуться к валидации.",
        "Запустите новую задачу: /task fsm start <имя>",
    ),
    (TaskStage.DONE, TaskStage.PAUSED): (
        "Задача завершена, пауза невозможна.",
        "Запустите новую задачу: /task fsm start <имя>",
    ),
    (TaskStage.PAUSED, TaskStage.DONE): (
        "Нельзя завершить задачу напрямую из паузы.",
        "Сначала возобновите задачу: /task fsm resume, затем продолжайте по цепочке.",
    ),
}


class ForbiddenTransitionError(ValueError):
    """Исключение при попытке выполнить недопустимый переход между этапами FSM."""

    def __init__(
        self,
        from_stage: TaskStage,
        to_stage: TaskStage,
        reason: str = "",
        hint: str = "",
    ) -> None:
        self.from_stage = from_stage
        self.to_stage = to_stage
        self.reason = reason or "Переход не разрешён."
        self.hint = hint or "Используйте /task fsm next для последовательного перехода."
        from_label = STAGE_LABELS.get(from_stage, from_stage.value)
        to_label = STAGE_LABELS.get(to_stage, to_stage.value)
        message = (
            f"Недопустимый переход: «{from_label}» → «{to_label}».\n"
            f"{self.reason}\n"
            f"Подсказка: {self.hint}"
        )
        super().__init__(message)

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

    def goto(self, target: TaskStage, note: str = "") -> None:
        """Явный переход в указанный этап — всегда блокируется для прямых переходов.

        Метод намеренно запрещает прямое указание целевого этапа пользователем:
        переходы выполняются только через advance() (next), pause() и resume().
        Любая попытка goto() выбрасывает ForbiddenTransitionError с объяснением.

        Используется командой /task fsm goto для демонстрации защиты FSM.
        """
        if self.stage == target:
            raise ForbiddenTransitionError(
                self.stage,
                target,
                reason=f"Задача уже находится на этапе «{STAGE_LABELS.get(target, target.value)}».",
                hint="Используйте /task fsm next для перехода к следующему этапу.",
            )

        pair = (self.stage, target)
        reason, hint = TRANSITION_BLOCK_REASONS.get(pair, (
            "Прямые переходы через goto запрещены.",
            "Используйте /task fsm next, /task fsm pause или /task fsm resume.",
        ))
        raise ForbiddenTransitionError(self.stage, target, reason=reason, hint=hint)

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
