"""MCP-сервер для CRM поддержки пользователей (stdio-транспорт).

Инструменты:
- get_user(user_id)            — данные пользователя
- get_ticket(ticket_id)        — тикет + данные пользователя
- list_tickets(status, user_id) — список тикетов с фильтрацией
- update_ticket_status(...)    — обновить статус тикета
"""

from __future__ import annotations

import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("support-crm")

_PROJECT_ROOT = Path(__file__).parents[2]
_CRM_DIR = _PROJECT_ROOT / "data" / "crm"
_USERS_FILE = _CRM_DIR / "users.json"
_TICKETS_FILE = _CRM_DIR / "tickets.json"


def _load_users() -> list[dict]:
    return json.loads(_USERS_FILE.read_text(encoding="utf-8"))


def _load_tickets() -> list[dict]:
    return json.loads(_TICKETS_FILE.read_text(encoding="utf-8"))


def _save_tickets(tickets: list[dict]) -> None:
    _TICKETS_FILE.write_text(json.dumps(tickets, ensure_ascii=False, indent=2), encoding="utf-8")


@mcp.tool()
def get_user(user_id: str) -> str:
    """Получить данные пользователя по его ID.

    Возвращает имя, email, тарифный план, дату регистрации и компанию.
    user_id — идентификатор пользователя, например: usr_001
    """
    users = _load_users()
    user = next((u for u in users if u["id"] == user_id), None)
    if not user:
        return f"Пользователь с ID '{user_id}' не найден."

    company = user.get("company") or "—"
    return (
        f"Пользователь: {user['name']}\n"
        f"  ID:          {user['id']}\n"
        f"  Email:       {user['email']}\n"
        f"  Тариф:       {user['plan'].upper()}\n"
        f"  Компания:    {company}\n"
        f"  Телефон:     {user.get('phone', '—')}\n"
        f"  Зарегистрирован: {user['registered_at']}\n"
        f"  Часовой пояс: {user.get('timezone', '—')}"
    )


@mcp.tool()
def get_ticket(ticket_id: str) -> str:
    """Получить полную информацию о тикете по его ID, включая данные пользователя.

    Возвращает тему, описание, тип, статус, приоритет и контекст пользователя.
    ticket_id — идентификатор тикета, например: TKT-001
    """
    tickets = _load_tickets()
    ticket = next((t for t in tickets if t["id"] == ticket_id), None)
    if not ticket:
        return f"Тикет '{ticket_id}' не найден."

    users = _load_users()
    user = next((u for u in users if u["id"] == ticket["user_id"]), None)
    user_info = (
        f"{user['name']} ({user['email']}, тариф: {user['plan'].upper()})"
        if user
        else f"ID: {ticket['user_id']} (не найден)"
    )

    resolution = ticket.get("resolution", "")
    resolution_line = f"\n  Решение:     {resolution}" if resolution else ""

    tags = ", ".join(ticket.get("tags", []))
    tags_line = f"\n  Теги:        {tags}" if tags else ""

    return (
        f"Тикет: {ticket['id']}\n"
        f"  Пользователь: {user_info}\n"
        f"  Тема:        {ticket['subject']}\n"
        f"  Тип:         {ticket['type']}\n"
        f"  Статус:      {ticket['status']}\n"
        f"  Приоритет:   {ticket['priority']}\n"
        f"  Создан:      {ticket['created_at']}\n"
        f"  Обновлён:    {ticket['updated_at']}"
        f"{tags_line}"
        f"\n\n  Описание:\n  {ticket['description']}"
        f"{resolution_line}"
    )


@mcp.tool()
def list_tickets(status: str = "", user_id: str = "") -> str:
    """Получить список тикетов с опциональной фильтрацией.

    status  — фильтр по статусу: open, in_progress, resolved (пусто = все)
    user_id — фильтр по ID пользователя (пусто = все)

    Возвращает таблицу тикетов с ID, темой, типом, статусом и пользователем.
    """
    tickets = _load_tickets()
    users = _load_users()
    user_map = {u["id"]: u for u in users}

    if status:
        tickets = [t for t in tickets if t["status"] == status]
    if user_id:
        tickets = [t for t in tickets if t["user_id"] == user_id]

    if not tickets:
        filters = []
        if status:
            filters.append(f"статус={status}")
        if user_id:
            filters.append(f"user_id={user_id}")
        filter_str = ", ".join(filters) if filters else "нет фильтров"
        return f"Тикетов не найдено ({filter_str})."

    lines = [f"Тикеты ({len(tickets)} шт.):\n"]
    for t in tickets:
        user = user_map.get(t["user_id"])
        user_name = user["name"] if user else t["user_id"]
        plan = f" [{user['plan'].upper()}]" if user else ""
        priority_mark = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(t["priority"], "⚪")
        status_label = {"open": "открыт", "in_progress": "в работе", "resolved": "решён"}.get(
            t["status"], t["status"]
        )
        lines.append(
            f"  {priority_mark} {t['id']} [{status_label}] — {t['subject']}\n"
            f"     Тип: {t['type']} | Пользователь: {user_name}{plan}"
        )

    return "\n".join(lines)


@mcp.tool()
def update_ticket_status(ticket_id: str, status: str) -> str:
    """Обновить статус тикета.

    ticket_id — идентификатор тикета, например: TKT-001
    status    — новый статус: open, in_progress, resolved
    """
    allowed = {"open", "in_progress", "resolved"}
    if status not in allowed:
        return f"Недопустимый статус '{status}'. Допустимые значения: {', '.join(sorted(allowed))}"

    tickets = _load_tickets()
    ticket = next((t for t in tickets if t["id"] == ticket_id), None)
    if not ticket:
        return f"Тикет '{ticket_id}' не найден."

    old_status = ticket["status"]
    if old_status == status:
        return f"Тикет {ticket_id} уже имеет статус '{status}'."

    from datetime import datetime, timezone

    ticket["status"] = status
    ticket["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    _save_tickets(tickets)

    status_label = {"open": "открыт", "in_progress": "в работе", "resolved": "решён"}
    return (
        f"Тикет {ticket_id} обновлён:\n"
        f"  {status_label.get(old_status, old_status)} → {status_label.get(status, status)}\n"
        f"  Тема: {ticket['subject']}"
    )


if __name__ == "__main__":
    mcp.run()
