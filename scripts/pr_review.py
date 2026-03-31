"""AI Code Review script.

Получает diff из PR (через GitHub API или локальный файл в режиме SIMULATE=1),
выполняет RAG-поиск по документации проекта, вызывает LLM и публикует
структурированный комментарий с ревью.

Переменные окружения:
    OPENROUTER_API_KEY  — ключ OpenRouter (не нужен при USE_LOCAL=1)
    GITHUB_TOKEN        — токен GitHub для публикации комментария
    GITHUB_REPOSITORY   — repo в формате owner/repo
    PR_NUMBER           — номер PR
    SIMULATE            — если "1", читает локальный файл вместо GitHub API
    SIMULATE_FILE       — путь к файлу для симуляции (по умолчанию buggy_calc.py)
    REVIEW_MODEL        — ID модели (по умолчанию google/gemma-2-9b-it или "local")
    USE_RAG             — если "0", отключить RAG-поиск (по умолчанию "1")
    USE_LOCAL           — если "1", использовать локальный llama.cpp вместо OpenRouter
    LOCAL_URL           — base URL локального сервера (по умолчанию http://127.0.0.1:8081/v1/chat/completions)
"""
from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

import httpx

_PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from llm_cli.agent import Agent
from llm_cli.api import OpenRouterClient
from llm_cli.models import DEFAULT_MODEL, LOCAL_BASE_URL, LOCAL_MODEL_ID, OPENROUTER_URL

_REVIEW_SYSTEM_PROMPT = textwrap.dedent("""\
    Ты — опытный senior-разработчик, проводящий код-ревью.
    Тебе будет предоставлен git diff изменений в PR.
    Твоя задача — дать детальный, конструктивный ревью.

    Формат ответа (строго следуй структуре):

    ## Потенциальные баги
    Перечисли конкретные баги, ошибки, граничные случаи, проблемы с обработкой ошибок.
    Ссылайся на конкретные строки/функции из diff. Если багов нет — напиши "Не обнаружено".

    ## Архитектурные проблемы
    Опиши нарушения принципов SOLID, избыточную связность, неправильные абстракции,
    дублирование кода, проблемы с именованием. Если проблем нет — напиши "Не обнаружено".

    ## Рекомендации
    Дай конкретные улучшения: тесты, рефакторинг, документация, оптимизации,
    альтернативные подходы. Используй конкретные примеры кода если нужно.

    Отвечай на русском языке. Технические термины (функции, переменные, классы) оставляй как есть.
""")

_MAX_DIFF_CHARS = 12_000
_RAG_TOP_K = 4


def _get_api_key(use_local: bool) -> str:
    """Вернуть API-ключ. Для локального сервера ключ не нужен — используем заглушку."""
    if use_local:
        return os.environ.get("OPENROUTER_API_KEY", "local")
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        print("ERROR: OPENROUTER_API_KEY not set (set USE_LOCAL=1 to use local llama.cpp)", file=sys.stderr)
        sys.exit(1)
    return key


def _fetch_pr_diff_from_github(token: str, repo: str, pr_number: str) -> tuple[str, list[str]]:
    """Получить diff и список изменённых файлов через GitHub REST API."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3.diff",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    base = f"https://api.github.com/repos/{repo}"

    with httpx.Client(timeout=30.0) as client:
        diff_resp = client.get(f"{base}/pulls/{pr_number}", headers=headers)
        diff_resp.raise_for_status()
        diff_text = diff_resp.text

        files_headers = {**headers, "Accept": "application/vnd.github+json"}
        files_resp = client.get(f"{base}/pulls/{pr_number}/files", headers=files_headers)
        files_resp.raise_for_status()
        changed_files = [f["filename"] for f in files_resp.json()]

    return diff_text, changed_files


def _build_simulate_diff(file_path: Path) -> tuple[str, list[str]]:
    """Построить искусственный diff из локального файла для демо."""
    code = file_path.read_text(encoding="utf-8")
    lines = code.splitlines()
    diff_lines = [
        f"diff --git a/{file_path.name} b/{file_path.name}",
        f"--- a/{file_path.name}",
        f"+++ b/{file_path.name}",
        f"@@ -0,0 +1,{len(lines)} @@",
    ]
    diff_lines += [f"+{line}" for line in lines]
    return "\n".join(diff_lines), [str(file_path.name)]


def _rag_search(query: str, top_k: int = _RAG_TOP_K) -> str:
    """Поиск по FAISS-индексу проекта. Возвращает контекст или пустую строку."""
    try:
        from llm_cli.rag.embedder import Embedder
        from llm_cli.rag.indexer import FaissIndex

        index_dir = _PROJECT_ROOT / "data" / "index" / "fixed"
        if not (index_dir / "index.faiss").exists():
            return ""

        embedder = Embedder(silent=True)
        index = FaissIndex.load(index_dir, dim=embedder.dim)
        query_emb = embedder.encode([query], show_progress=False)[0]
        results = index.search(query_emb, top_k=top_k)

        if not results:
            return ""

        parts = []
        for chunk, score in results:
            if score < 0.2:
                continue
            src = chunk.metadata.get("file", chunk.metadata.get("source", ""))
            parts.append(f"[{src}] {chunk.text.strip()}")

        return "\n\n".join(parts)
    except Exception as exc:  # noqa: BLE001
        print(f"RAG search skipped: {exc}", file=sys.stderr)
        return ""


def _build_review_prompt(diff: str, changed_files: list[str], rag_context: str) -> str:
    """Собрать промпт для LLM из diff и RAG-контекста."""
    truncated = diff[:_MAX_DIFF_CHARS]
    if len(diff) > _MAX_DIFF_CHARS:
        truncated += "\n\n[... diff truncated ...]"

    files_str = ", ".join(changed_files) if changed_files else "неизвестно"

    parts = [
        f"Изменённые файлы: {files_str}",
        "",
        "```diff",
        truncated,
        "```",
    ]

    if rag_context:
        parts += [
            "",
            "--- Контекст из документации проекта (RAG) ---",
            rag_context,
            "--- Конец контекста ---",
        ]

    return "\n".join(parts)


def _post_github_comment(token: str, repo: str, pr_number: str, body: str) -> None:
    """Опубликовать комментарий к PR через GitHub API."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(url, headers=headers, json={"body": body})
        resp.raise_for_status()
    print(f"Comment posted: {resp.json().get('html_url', 'ok')}")


def _format_comment(review: str, changed_files: list[str], used_rag: bool) -> str:
    """Обернуть ревью в markdown-шаблон для GitHub."""
    files_str = "\n".join(f"- `{f}`" for f in changed_files) if changed_files else "- (нет данных)"
    rag_note = " + RAG (документация проекта)" if used_rag else ""
    return textwrap.dedent(f"""\
        ## 🤖 AI Code Review{rag_note}

        **Изменённые файлы:**
        {files_str}

        ---

        {review}

        ---
        *Ревью сгенерировано автоматически с помощью LLM. Всегда проверяйте выводы вручную.*
    """)


def main() -> None:
    simulate = os.environ.get("SIMULATE", "0") == "1"
    use_rag = os.environ.get("USE_RAG", "1") != "0"
    use_local = os.environ.get("USE_LOCAL", "0") == "1"
    local_url = os.environ.get("LOCAL_URL", LOCAL_BASE_URL)

    if use_local:
        model = os.environ.get("REVIEW_MODEL", LOCAL_MODEL_ID)
        base_url = local_url
        print(f"Mode: LOCAL llama.cpp ({local_url}), model={model}")
    else:
        model = os.environ.get("REVIEW_MODEL", DEFAULT_MODEL)
        base_url = OPENROUTER_URL
        print(f"Mode: OpenRouter, model={model}")

    api_key = _get_api_key(use_local)

    if simulate:
        simulate_file = os.environ.get("SIMULATE_FILE", str(_PROJECT_ROOT / "buggy_calc.py"))
        file_path = Path(simulate_file)
        print(f"[SIMULATE] Reading local file: {file_path}")
        diff, changed_files = _build_simulate_diff(file_path)
    else:
        token = os.environ.get("GITHUB_TOKEN", "").strip()
        repo = os.environ.get("GITHUB_REPOSITORY", "").strip()
        pr_number = os.environ.get("PR_NUMBER", "").strip()

        if not token or not repo or not pr_number:
            print("ERROR: GITHUB_TOKEN, GITHUB_REPOSITORY and PR_NUMBER are required", file=sys.stderr)
            sys.exit(1)

        print(f"Fetching diff for PR #{pr_number} in {repo}...")
        diff, changed_files = _fetch_pr_diff_from_github(token, repo, pr_number)

    print(f"Changed files: {', '.join(changed_files)}")
    print(f"Diff size: {len(diff)} chars")

    rag_context = ""
    if use_rag:
        print("Running RAG search...")
        rag_query = f"code review architecture patterns: {' '.join(changed_files)}"
        rag_context = _rag_search(rag_query)
        if rag_context:
            print(f"RAG: found {len(rag_context)} chars of context")
        else:
            print("RAG: no relevant context found (index may be empty)")

    prompt = _build_review_prompt(diff, changed_files, rag_context)

    print(f"Calling LLM ({model})...")
    with OpenRouterClient(api_key, base_url=base_url) as client:
        agent = Agent(
            client=client,
            model=model,
            temperature=0.2,
            system_prompt=_REVIEW_SYSTEM_PROMPT,
            compression_enabled=False,
        )
        review = agent.run(prompt)

    comment_body = _format_comment(review, changed_files, used_rag=bool(rag_context))

    print("\n" + "=" * 60)
    print(comment_body)
    print("=" * 60 + "\n")

    if not simulate:
        token = os.environ.get("GITHUB_TOKEN", "").strip()
        repo = os.environ.get("GITHUB_REPOSITORY", "").strip()
        pr_number = os.environ.get("PR_NUMBER", "").strip()
        print("Publishing comment to GitHub...")
        _post_github_comment(token, repo, pr_number, comment_body)
    else:
        print("[SIMULATE] GitHub comment not published (SIMULATE=1)")


if __name__ == "__main__":
    main()
