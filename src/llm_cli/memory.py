"""3-слойная модель памяти ассистента: краткосрочная / рабочая / долговременная."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from platformdirs import user_config_dir
from pydantic import BaseModel, Field

from .task_fsm import ForbiddenTransitionError, TaskFSM, TaskStage

APP_NAME = "llm-cli"
LONG_TERM_FILENAME = "long_term_memory.json"

# Макс. количество решений и заметок в долговременной памяти
MAX_DECISIONS = 20
MAX_NOTES = 30


def _long_term_path() -> Path:
    config_dir = Path(user_config_dir(APP_NAME, appauthor=False, ensure_exists=True))
    return config_dir / LONG_TERM_FILENAME


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic-модели слоёв памяти
# ─────────────────────────────────────────────────────────────────────────────


class WorkingMemory(BaseModel):
    """Рабочая память: данные текущей задачи. Живёт в рамках одной задачи/сессии."""

    task: str = ""
    goals: list[str] = Field(default_factory=list)
    facts: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    task_fsm: TaskFSM | None = None

    def is_empty(self) -> bool:
        return not self.task and not self.goals and not self.facts and not self.notes


class DecisionEntry(BaseModel):
    """Одно решение с меткой времени в долговременной памяти."""

    timestamp: str
    text: str


# Допустимые значения полей профиля (для валидации и подсказок)
PROFILE_STYLE_OPTIONS = ("нейтральный", "краткий", "подробный", "формальный", "дружеский")
PROFILE_FORMAT_OPTIONS = ("markdown", "plain", "bullets")
PROFILE_EXPERTISE_OPTIONS = ("начинающий", "средний", "эксперт")


class UserProfile(BaseModel):
    """Структурированный профиль пользователя — хранит предпочтения для персонализации."""

    name: str = ""
    language: str = "русский"
    style: str = "нейтральный"
    format: str = "markdown"
    expertise: str = "средний"
    domain: str = ""
    constraints: list[str] = Field(default_factory=list)

    def build_system_prompt(self) -> str:
        """Сформировать системную инструкцию из профиля для инъекции в запрос."""
        lines = ["[ПРОФИЛЬ ПОЛЬЗОВАТЕЛЯ]"]

        if self.name:
            lines.append(f"Обращайся к пользователю по имени: {self.name}")

        lines.append(f"Язык ответов: {self.language}")

        style_map = {
            "краткий": "Стиль: краткий и по делу, без вступлений и лишних слов",
            "подробный": "Стиль: подробный, с объяснениями и примерами",
            "формальный": "Стиль: формальный, деловой тон",
            "дружеский": "Стиль: дружелюбный, неформальный",
            "нейтральный": "Стиль: нейтральный",
        }
        lines.append(style_map.get(self.style, f"Стиль: {self.style}"))

        format_map = {
            "markdown": "Формат: используй Markdown — заголовки, жирный текст, блоки кода",
            "plain": "Формат: обычный текст без Markdown-разметки",
            "bullets": "Формат: структурируй ответы через маркированные списки",
        }
        lines.append(format_map.get(self.format, f"Формат: {self.format}"))

        expertise_map = {
            "начинающий": "Уровень пользователя: начинающий — объясняй с нуля, используй аналогии, избегай жаргона",
            "средний": "Уровень пользователя: средний — предполагай базовые знания",
            "эксперт": "Уровень пользователя: эксперт — не объясняй очевидное, используй профессиональную терминологию",
        }
        lines.append(expertise_map.get(self.expertise, f"Уровень: {self.expertise}"))

        if self.domain:
            lines.append(f"Область работы пользователя: {self.domain}")

        if self.constraints:
            lines.append("Ограничения (строго соблюдай):")
            for c in self.constraints:
                lines.append(f"- {c}")

        return "\n".join(lines)

    def is_default(self) -> bool:
        return (
            not self.name
            and self.language == "русский"
            and self.style == "нейтральный"
            and self.format == "markdown"
            and self.expertise == "средний"
            and not self.domain
            and not self.constraints
        )


class LongTermMemory(BaseModel):
    """Долговременная память: профиль, решения, знания. Не очищается при /clear."""

    profile: dict[str, str] = Field(default_factory=dict)
    decisions: list[DecisionEntry] = Field(default_factory=list)
    knowledge: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    # Структурированные именованные профили пользователя
    profiles: dict[str, UserProfile] = Field(default_factory=dict)
    active_profile: str = ""

    def is_empty(self) -> bool:
        return (
            not self.profile
            and not self.decisions
            and not self.knowledge
            and not self.notes
        )


# ─────────────────────────────────────────────────────────────────────────────
# MemoryManager
# ─────────────────────────────────────────────────────────────────────────────


class MemoryManager:
    """Управляет тремя слоями памяти: рабочей и долговременной (краткосрочная — в агенте)."""

    def __init__(self) -> None:
        self._working = WorkingMemory()
        self._long_term = LongTermMemory()
        self._long_term_file = _long_term_path()
        self._load_long_term()

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def working(self) -> WorkingMemory:
        return self._working

    @property
    def long_term(self) -> LongTermMemory:
        return self._long_term

    # ── Рабочая память ───────────────────────────────────────────────────────

    def set_task(self, task: str) -> None:
        self._working.task = task.strip()

    def add_goal(self, goal: str) -> None:
        goal = goal.strip()
        if goal and goal not in self._working.goals:
            self._working.goals.append(goal)

    def set_working_fact(self, key: str, value: str) -> None:
        self._working.facts[key.strip()] = value.strip()

    def add_working_note(self, note: str) -> None:
        note = note.strip()
        if note:
            self._working.notes.append(note)

    def clear_working(self) -> None:
        self._working = WorkingMemory()

    def update_working_from_message(self, text: str) -> None:
        """Автоматически извлекает данные задачи из сообщения пользователя."""
        lowered = text.lower()

        # Задача
        if self._working.task == "":
            m = re.search(
                r"(?:задача|цель|task|goal)\s*[:\-]\s*(.+)", text, re.IGNORECASE
            )
            if m:
                self._working.task = m.group(1).strip()[:120]

        # Ключевые факты задачи (проект, дедлайн, стек — аналог sticky facts)
        kv_patterns = [
            (r"проект\s*[:\-]\s*(.+)", "проект"),
            (r"проект называется\s+(.+)", "проект"),
            (r"дедлайн\s*[:\-]\s*(.+)", "дедлайн"),
            (r"стек\s*[:\-]\s*(.+)", "стек"),
            (r"клиент\s*[:\-]\s*(.+)", "клиент"),
            (r"бюджет\s*[:\-]\s*(.+)", "бюджет"),
        ]
        for pattern, key in kv_patterns:
            if key not in self._working.facts:
                m = re.search(pattern, text, re.IGNORECASE)
                if m:
                    self._working.facts[key] = m.group(1).strip()[:90]

        # Цели / приоритеты
        if "приоритет" in lowered or "цель" in lowered:
            snippet = text.strip()[:100]
            if snippet not in self._working.goals and len(self._working.goals) < 5:
                self._working.goals.append(snippet)

    # ── Долговременная память ─────────────────────────────────────────────────

    def remember_knowledge(self, key: str, value: str) -> None:
        self._long_term.knowledge[key.strip()] = value.strip()
        self._touch_long_term()
        self._save_long_term()

    def remember_profile(self, key: str, value: str) -> None:
        self._long_term.profile[key.strip()] = value.strip()
        self._touch_long_term()
        self._save_long_term()

    def remember_decision(self, text: str) -> None:
        entry = DecisionEntry(
            timestamp=datetime.now().strftime("%Y-%m-%d"),
            text=text.strip(),
        )
        self._long_term.decisions.append(entry)
        if len(self._long_term.decisions) > MAX_DECISIONS:
            self._long_term.decisions = self._long_term.decisions[-MAX_DECISIONS:]
        self._touch_long_term()
        self._save_long_term()

    def remember_note(self, text: str) -> None:
        note = text.strip()
        if note:
            self._long_term.notes.append(note)
            if len(self._long_term.notes) > MAX_NOTES:
                self._long_term.notes = self._long_term.notes[-MAX_NOTES:]
            self._touch_long_term()
            self._save_long_term()

    def forget(self, key: str) -> bool:
        """Удалить ключ из knowledge или profile. Возвращает True если удалено."""
        key = key.strip()
        if key in self._long_term.knowledge:
            del self._long_term.knowledge[key]
            self._touch_long_term()
            self._save_long_term()
            return True
        if key in self._long_term.profile:
            del self._long_term.profile[key]
            self._touch_long_term()
            self._save_long_term()
            return True
        return False

    # ── Управление профилями пользователя ────────────────────────────────────

    def get_active_profile(self) -> UserProfile | None:
        name = self._long_term.active_profile
        if name and name in self._long_term.profiles:
            return self._long_term.profiles[name]
        return None

    def get_active_profile_name(self) -> str:
        return self._long_term.active_profile

    def list_profiles(self) -> list[str]:
        return list(self._long_term.profiles.keys())

    def save_profile(self, name: str, profile: UserProfile) -> None:
        name = name.strip()
        self._long_term.profiles[name] = profile
        self._touch_long_term()
        self._save_long_term()

    def switch_profile(self, name: str) -> None:
        name = name.strip()
        if name not in self._long_term.profiles:
            raise ValueError(f"Профиль «{name}» не найден. Доступные: {self.list_profiles()}")
        self._long_term.active_profile = name
        self._touch_long_term()
        self._save_long_term()

    def deactivate_profile(self) -> None:
        """Отключить активный профиль (вернуться к дефолтному поведению)."""
        self._long_term.active_profile = ""
        self._touch_long_term()
        self._save_long_term()

    def delete_profile(self, name: str) -> bool:
        name = name.strip()
        if name not in self._long_term.profiles:
            return False
        del self._long_term.profiles[name]
        if self._long_term.active_profile == name:
            self._long_term.active_profile = ""
        self._touch_long_term()
        self._save_long_term()
        return True

    def set_profile_field(self, name: str, field: str, value: str) -> bool:
        """Установить поле существующего профиля. Возвращает True если поле известно."""
        name = name.strip()
        if name not in self._long_term.profiles:
            raise ValueError(f"Профиль «{name}» не найден.")
        profile = self._long_term.profiles[name]
        field = field.strip().lower()
        value = value.strip()
        known_fields = {"name", "language", "style", "format", "expertise", "domain"}
        if field in known_fields:
            updated = profile.model_dump()
            updated[field] = value
            self._long_term.profiles[name] = UserProfile.model_validate(updated)
            self._touch_long_term()
            self._save_long_term()
            return True
        if field == "constraint":
            profile.constraints.append(value)
            self._touch_long_term()
            self._save_long_term()
            return True
        return False

    def build_profile_block(self) -> str:
        """Сформировать системный блок из активного профиля для инъекции в запрос."""
        profile = self.get_active_profile()
        if profile is None or profile.is_default():
            return ""
        return profile.build_system_prompt()

    # ── Управление FSM задачи ─────────────────────────────────────────────────

    def start_fsm(self, task_name: str) -> TaskFSM:
        """Запустить новый FSM в этапе planning."""
        fsm = TaskFSM(
            task_name=task_name.strip(),
            stage=TaskStage.PLANNING,
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        )
        self._working.task_fsm = fsm
        return fsm

    def get_fsm(self) -> TaskFSM | None:
        """Вернуть текущий FSM или None если не активен."""
        return self._working.task_fsm

    def advance_fsm(self, note: str = "") -> TaskFSM:
        """Перейти к следующему этапу FSM."""
        fsm = self._require_fsm()
        fsm.advance(note)
        return fsm

    def pause_fsm(self) -> TaskFSM:
        """Поставить FSM на паузу."""
        fsm = self._require_fsm()
        fsm.pause()
        return fsm

    def resume_fsm(self) -> TaskFSM:
        """Возобновить FSM с сохранённого этапа."""
        fsm = self._require_fsm()
        fsm.resume()
        return fsm

    def goto_fsm(self, target: TaskStage, note: str = "") -> TaskFSM:
        """Явный переход FSM в указанный этап с проверкой допустимости.

        Выбрасывает ForbiddenTransitionError если переход не разрешён.
        """
        fsm = self._require_fsm()
        fsm.goto(target, note)
        return fsm

    def add_fsm_artifact(self, key: str, text: str) -> TaskFSM:
        """Добавить артефакт к текущему этапу FSM."""
        fsm = self._require_fsm()
        fsm.add_artifact(key, text)
        return fsm

    def set_fsm_step(self, step: str, expected_action: str = "") -> None:
        """Установить текущий шаг и ожидаемое действие внутри этапа FSM."""
        fsm = self._require_fsm()
        fsm.set_step(step, expected_action)

    def clear_fsm(self) -> None:
        """Сбросить FSM (удалить из рабочей памяти)."""
        self._working.task_fsm = None

    def _require_fsm(self) -> TaskFSM:
        if self._working.task_fsm is None:
            raise ValueError(
                "FSM не активен. Запустите задачу командой: /task fsm start <имя>"
            )
        return self._working.task_fsm

    # ── Инъекция в запрос ─────────────────────────────────────────────────────

    def build_context_block(self) -> str:
        """Собрать блок-инъекцию памяти для системного сообщения."""
        lines: list[str] = []

        # Рабочая память
        if not self._working.is_empty():
            lines.append("=== Рабочая память (текущая задача) ===")
            if self._working.task:
                lines.append(f"Задача: {self._working.task}")
            if self._working.facts:
                facts_str = ", ".join(
                    f"{k}={v}" for k, v in self._working.facts.items()
                )
                lines.append(f"Факты: {facts_str}")
            if self._working.goals:
                for i, g in enumerate(self._working.goals, 1):
                    lines.append(f"Цель {i}: {g}")
            if self._working.notes:
                for note in self._working.notes[-3:]:
                    lines.append(f"Заметка: {note}")

        # Долговременная память
        if not self._long_term.is_empty():
            if lines:
                lines.append("")
            lines.append("=== Долговременная память ===")
            if self._long_term.profile:
                profile_str = ", ".join(
                    f"{k}={v}" for k, v in self._long_term.profile.items()
                )
                lines.append(f"Профиль: {profile_str}")
            if self._long_term.knowledge:
                for k, v in self._long_term.knowledge.items():
                    lines.append(f"Знание [{k}]: {v}")
            if self._long_term.decisions:
                for d in self._long_term.decisions[-5:]:
                    lines.append(f"Решение [{d.timestamp}]: {d.text}")
            if self._long_term.notes:
                for note in self._long_term.notes[-3:]:
                    lines.append(f"Заметка: {note}")

        # FSM-статус задачи
        if self._working.task_fsm is not None:
            if lines:
                lines.append("")
            lines.append(self._working.task_fsm.build_status_block())

        if not lines:
            return ""

        return "[КОНТЕКСТ ПАМЯТИ АССИСТЕНТА]\n" + "\n".join(lines)

    # ── Сериализация рабочей памяти (для history.json) ────────────────────────

    def working_to_dict(self) -> dict[str, object]:
        return self._working.model_dump()

    def working_from_dict(self, data: dict[str, object]) -> None:
        try:
            self._working = WorkingMemory.model_validate(data)
        except Exception:
            self._working = WorkingMemory()

    # ── Долговременная память: хранилище ─────────────────────────────────────

    def _touch_long_term(self) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        if not self._long_term.created_at:
            self._long_term.created_at = now
        self._long_term.updated_at = now

    def _load_long_term(self) -> None:
        if not self._long_term_file.exists():
            return
        try:
            raw = self._long_term_file.read_text(encoding="utf-8")
            if not raw.strip():
                return
            payload = json.loads(raw)
            if isinstance(payload, dict):
                self._long_term = LongTermMemory.model_validate(payload)
        except (OSError, json.JSONDecodeError, Exception):
            return

    def _save_long_term(self) -> None:
        try:
            self._long_term_file.parent.mkdir(parents=True, exist_ok=True)
            self._long_term_file.write_text(
                json.dumps(self._long_term.model_dump(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            return
