"""AI Daily Digest — ежедневный дайджест активности репозитория.

Собирает git-активность за последние 24 часа, TODO/FIXME из кода,
отправляет в LLM и выводит структурированный отчёт через Rich.

Точка входа: run_daily_digest(client, model, repo_path)
"""

from __future__ import annotations

import subprocess
import textwrap
from datetime import datetime
from pathlib import Path

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.live import Live

console = Console()

_SYSTEM_PROMPT = textwrap.dedent("""\
    Ты — опытный техлид, который каждое утро готовит дайджест для команды.
    Тебе дана информация об активности в git-репозитории за последние 24 часа.

    Структура ответа (строго следуй, используй markdown):

    ## Что сделано
    Краткое резюме коммитов. Сгруппируй по смыслу, не перечисляй дословно.
    Если коммитов не было — напиши "Вчера коммитов не было".

    ## Ключевые изменения
    Самые значимые изменённые файлы/модули и что в них поменялось.
    Пропусти если коммитов не было.

    ## Риски и внимание
    Что требует внимания: большие diff, TODO/FIXME в коде, опасные места.
    Если рисков нет — напиши "Явных рисков не обнаружено".

    ## Фокус на сегодня
    3-5 конкретных рекомендаций: что стоит сделать сегодня исходя из состояния репозитория.

    Отвечай на русском языке. Будь конкретным, не лей воду.
""")

_MAX_COMMIT_LINES = 60
_MAX_TODO_LINES = 40


def _run(cmd: list[str], cwd: Path) -> str:
    """Запустить команду и вернуть stdout. При ошибке — пустую строку."""
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""


def collect_git_activity(repo_path: Path, since_hours: int = 24) -> dict[str, str]:
    """Собрать git-активность: коммиты, diff-статистика, авторы."""
    since = f"{since_hours} hours ago"

    commits = _run(
        [
            "git", "log",
            f"--since={since}",
            "--pretty=format:%h  %an  %s",
            "--no-merges",
        ],
        repo_path,
    )

    diff_stat = _run(
        [
            "git", "diff",
            f"--since={since}",
            "--stat",
            f"HEAD@{{24 hours ago}}",
            "HEAD",
        ],
        repo_path,
    )

    if not diff_stat:
        diff_stat = _run(
            ["git", "diff", "--stat", "HEAD~1", "HEAD"],
            repo_path,
        )

    authors_raw = _run(
        [
            "git", "log",
            f"--since={since}",
            "--no-merges",
            "--pretty=format:%an",
        ],
        repo_path,
    )
    authors = list(dict.fromkeys(authors_raw.splitlines())) if authors_raw else []

    commit_count_raw = _run(
        ["git", "rev-list", "--count", f"--since={since}", "HEAD"],
        repo_path,
    )
    commit_count = int(commit_count_raw) if commit_count_raw.isdigit() else 0

    commits_trimmed = "\n".join(commits.splitlines()[:_MAX_COMMIT_LINES])
    if len(commits.splitlines()) > _MAX_COMMIT_LINES:
        commits_trimmed += f"\n... (ещё {len(commits.splitlines()) - _MAX_COMMIT_LINES} коммитов)"

    return {
        "commits": commits_trimmed,
        "diff_stat": diff_stat[:3000] if diff_stat else "",
        "authors": ", ".join(authors) if authors else "нет",
        "commit_count": str(commit_count),
    }


def collect_todos(repo_path: Path) -> dict[str, int | str]:
    """Найти TODO/FIXME/HACK в кодовой базе через rg или grep (исключает .venv)."""
    def _search(pattern: str) -> list[str]:
        out = _run(
            [
                "rg",
                "--glob=*.py",
                "--glob=!.venv/**",
                "--glob=!venv/**",
                "--glob=!node_modules/**",
                "-n", "--no-heading", "-i",
                pattern,
            ],
            repo_path,
        )
        if not out:
            out = _run(
                [
                    "grep", "-rn",
                    "--include=*.py",
                    "--exclude-dir=.venv",
                    "--exclude-dir=venv",
                    "--exclude-dir=node_modules",
                    "-i", pattern, ".",
                ],
                repo_path,
            )
        return [line for line in out.splitlines() if line.strip()][:_MAX_TODO_LINES]

    todos = _search("TODO")
    fixmes = _search("FIXME")
    hacks = _search("HACK")

    sample_lines: list[str] = []
    for label, items in [("TODO", todos), ("FIXME", fixmes), ("HACK", hacks)]:
        for line in items[:5]:
            sample_lines.append(f"[{label}] {line}")

    return {
        "todo_count": len(todos),
        "fixme_count": len(fixmes),
        "hack_count": len(hacks),
        "sample": "\n".join(sample_lines[:20]),
    }


def build_digest_prompt(
    activity: dict[str, str],
    todos: dict[str, int | str],
    repo_path: Path,
) -> str:
    """Собрать промпт для LLM из собранных данных."""
    now = datetime.now().strftime("%A, %d %B %Y, %H:%M")
    repo_name = repo_path.resolve().name

    parts = [
        f"Репозиторий: {repo_name}",
        f"Дата и время: {now}",
        f"Период: последние 24 часа",
        "",
        f"## Коммиты ({activity['commit_count']} шт., авторы: {activity['authors']})",
    ]

    if activity["commits"]:
        parts.append(activity["commits"])
    else:
        parts.append("Коммитов за 24 часа не было.")

    if activity["diff_stat"]:
        parts += ["", "## Статистика изменений (git diff --stat)", activity["diff_stat"]]

    todo_total = todos["todo_count"] + todos["fixme_count"] + todos["hack_count"]
    parts += [
        "",
        f"## Технический долг в коде",
        f"TODO: {todos['todo_count']}, FIXME: {todos['fixme_count']}, HACK: {todos['hack_count']} (итого: {todo_total})",
    ]

    if todos["sample"]:
        parts += ["", "Примеры (первые 20):", str(todos["sample"])]

    return "\n".join(parts)


def _render_input_table(activity: dict[str, str], todos: dict[str, int | str]) -> None:
    """Показать таблицу с входными данными перед запросом к LLM."""
    table = Table(
        title="Данные для дайджеста",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Параметр", style="bold", width=24)
    table.add_column("Значение")

    table.add_row("Коммитов за 24ч", activity["commit_count"])
    table.add_row("Авторы", activity["authors"])
    table.add_row(
        "TODO / FIXME / HACK",
        f"{todos['todo_count']} / {todos['fixme_count']} / {todos['hack_count']}",
    )
    if activity["diff_stat"]:
        first_stat_line = activity["diff_stat"].splitlines()[-1] if activity["diff_stat"] else "—"
        table.add_row("Diff итог", first_stat_line)

    console.print(table)


def run_daily_digest(
    client: object,
    model: str,
    repo_path: Path = Path("."),
) -> None:
    """Основная функция: собрать данные, вызвать LLM, вывести дайджест."""
    from .agent import Agent
    from .models import ChatMessage

    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]AI Daily Digest[/bold cyan]\n"
            "[dim]Ежедневный дайджест активности репозитория[/dim]",
            border_style="cyan",
        )
    )
    console.print()

    with console.status("[cyan]Собираю git-активность...[/cyan]"):
        activity = collect_git_activity(repo_path)

    with console.status("[cyan]Ищу TODO/FIXME в коде...[/cyan]"):
        todos = collect_todos(repo_path)

    _render_input_table(activity, todos)
    console.print()

    if activity["commits"]:
        console.print(
            Panel(
                activity["commits"],
                title="[bold]Коммиты за 24ч[/bold]",
                border_style="dim",
                expand=False,
            )
        )
        console.print()

    prompt = build_digest_prompt(activity, todos, repo_path)

    console.print(Rule("[cyan]Запрос к LLM[/cyan]", style="cyan"))
    console.print(f"[dim]Модель: {model}[/dim]")
    console.print()

    digest_text = ""
    with console.status(f"[cyan]AI анализирует репозиторий...[/cyan]"):
        agent = Agent(
            client=client,
            model=model,
            temperature=0.3,
            system_prompt=_SYSTEM_PROMPT,
            compression_enabled=False,
        )
        digest_text = agent.run(prompt)

    console.print(Rule("[bold green]AI Daily Digest[/bold green]", style="green"))
    console.print()
    console.print(Markdown(digest_text))
    console.print()
    console.print(
        Panel.fit(
            f"[dim]Сгенерировано: {datetime.now().strftime('%d.%m.%Y %H:%M')} · модель: {model}[/dim]",
            border_style="dim",
        )
    )
    console.print()
