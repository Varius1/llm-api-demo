"""
Тесты для search_history_db.py и execution_logger.py

Демонстрируют работу с:
- SQLite базой данных (история поиска)
- JSON Lines логированием (выполнение команд)
- Семантическим поиском
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import sqlite3
import json
import time


class TestSearchHistoryDB:
    """Тесты для search_history_db.py"""

    @pytest.fixture
    def mock_db_path(self, tmp_path, monkeypatch):
        """Mock для пути к БД"""
        with patch("llm_cli.search_history_db._get_db_path", return_value=tmp_path / "search_history.db"):
            yield tmp_path

    def test_init_db_creates_tables(self, mock_db_path):
        """Инициализация создаёт таблицы"""
        from llm_cli.search_history_db import init_db

        conn = init_db()
        
        # Проверяем существование таблиц
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        )
        tables = [row[0] for row in cursor.fetchall()]

        assert "search_history" in tables
        assert "search_history_vec" in tables

        conn.close()

    def test_save_and_retrieve_search(self, mock_db_path):
        """Сохранение и получение поиска"""
        from llm_cli.search_history_db import init_db, save_search, search_history

        conn = init_db()

        # Сохраняем результат поиска
        results = {"results": ["result1", "result2"]}
        summary = "Test search summary"

        save_search(
            conn,
            session_id="session_123",
            query="test query",
            results=results,
            summary=summary
        )
        conn.close()

        # Загружаем и ищем
        conn = init_db()
        found = search_history(conn, "test", session_id="session_123", limit=10)

        assert len(found) > 0
        assert found[0]["query"] == "test query"
        conn.close()

    def test_session_filtering(self, mock_db_path):
        """Фильтрация по сессии"""
        from llm_cli.search_history_db import init_db, save_search, list_recent_searches

        conn = init_db()

        # Сохраняем поиски для разных сессий
        save_search(conn, "session_1", "query1", {}, "summary1")
        save_search(conn, "session_2", "query2", {}, "summary2")
        save_search(conn, "session_1", "query3", {}, "summary3")

        conn.close()

        # Загружаем только для session_1
        conn = init_db()
        session1_searches = list_recent_searches(conn, session_id="session_1", limit=10)

        assert len(session1_searches) == 2
        queries = [s["query"] for s in session1_searches]
        assert "query1" in queries
        assert "query3" in queries
        assert "query2" not in queries
        conn.close()

    def test_duplicate_prevention(self, mock_db_path):
        """Предотвращение дубликатов"""
        from llm_cli.search_history_db import init_db, save_search, list_recent_searches

        conn = init_db()

        # Сохраняем одинаковый поиск дважды
        save_search(conn, "session_1", "same query", {"data": 1}, "summary")
        save_search(conn, "session_1", "same query", {"data": 1}, "summary")

        conn.close()

        # Проверяем, что только одна запись
        conn = init_db()
        searches = list_recent_searches(conn, session_id="session_1", limit=10)
        conn.close()

        assert len(searches) == 1

    def test_clear_history(self, mock_db_path):
        """Очистка истории"""
        from llm_cli.search_history_db import init_db, save_search, list_recent_searches, clear_search_history

        conn = init_db()

        # Добавляем записи
        save_search(conn, "session_1", "query1", {}, "summary1")
        save_search(conn, "session_1", "query2", {}, "summary2")

        conn.close()

        # Очищаем
        conn = init_db()
        clear_search_history(conn, session_id="session_1")
        searches = list_recent_searches(conn, session_id="session_1", limit=10)
        conn.close()

        assert len(searches) == 0

    def test_get_search_by_id(self, mock_db_path):
        """Получение поиска по ID"""
        from llm_cli.search_history_db import init_db, save_search, get_search_by_id

        conn = init_db()

        save_search(conn, "session_1", "test query", {"key": "value"}, "test summary")
        conn.close()

        # Получаем первую запись
        conn = init_db()
        search = get_search_by_id(conn, 1)
        conn.close()

        assert search is not None
        assert search["query"] == "test query"

    @pytest.mark.skip(reason="Требует SentenceTransformer, может быть медленным")
    def test_semantic_search(self, mock_db_path):
        """Семантический поиск по эмбеддингам"""
        from llm_cli.search_history_db import init_db, save_search, search_history

        conn = init_db()

        # Сохраняем поиски
        save_search(conn, "session_1", "python programming", {}, "about python")
        save_search(conn, "session_1", "machine learning", {}, "about ml")

        conn.close()

        # Ищем семантически похожие
        conn = init_db()
        results = search_history(conn, "coding in python", session_id="session_1", limit=10)
        conn.close()

        assert len(results) > 0


class TestExecutionLogger:
    """Тесты для execution_logger.py"""

    @pytest.fixture
    def mock_log_path(self, tmp_path, monkeypatch):
        """Mock для пути к логам"""
        log_file = tmp_path / "execution.log"
        
        with patch("llm_cli.execution_logger.LOG_FILE", log_file):
            yield log_file

    def test_log_command_execution(self, mock_log_path):
        """Логирование выполнения команды"""
        from llm_cli.execution_logger import log_command_execution, get_execution_logs

        # Логируем команду
        log_command_execution(
            cmd="ls -la",
            exit_code=0,
            duration_ms=123,
            user="agent",
            blocked=False
        )

        # Проверяем лог (возвращает строку)
        logs = get_execution_logs(limit=10)

        assert "ls -la" in logs
        assert "agent" in logs
        assert "0" in logs  # exit_code

    def test_log_blocked_command(self, mock_log_path):
        """Логирование заблокированной команды"""
        from llm_cli.execution_logger import log_command_execution, get_execution_logs

        log_command_execution(
            cmd="rm -rf /",
            exit_code=None,
            duration_ms=0,
            user="agent",
            blocked=True,
            block_reason="blocked pattern"
        )

        logs = get_execution_logs(limit=10)

        assert "blocked pattern" in logs
        assert "BLOCKED" in logs
        assert "agent" in logs

    def test_log_python_execution(self, mock_log_path):
        """Логирование выполнения Python кода"""
        from llm_cli.execution_logger import log_python_execution, get_execution_logs

        log_python_execution(
            code_hash="abc123",
            success=True,
            duration_ms=456,
            user="agent"
        )

        logs = get_execution_logs(limit=10)

        assert "Python" in logs
        assert "agent" in logs
        assert "456ms" in logs
        assert "✅" in logs  # Иконка успеха

    def test_log_python_execution_error(self, mock_log_path):
        """Логирование ошибки Python"""
        from llm_cli.execution_logger import log_python_execution, get_execution_logs

        log_python_execution(
            code_hash="def456",
            success=False,
            duration_ms=100,
            user="agent",
            error="SyntaxError: invalid syntax"
        )

        logs = get_execution_logs(limit=10)

        assert "Python" in logs
        assert "SyntaxError" in logs
        assert "❌" in logs  # Иконка ошибки
        assert "100ms" in logs

    def test_get_logs_filtering(self, mock_log_path):
        """Фильтрация логов"""
        from llm_cli.execution_logger import log_command_execution, get_execution_logs

        # Логируем разные команды
        log_command_execution("cmd1", 0, 100, "user1", False)
        log_command_execution("cmd2", 0, 100, "user2", False)
        log_command_execution("cmd3", 1, 100, "user1", False)

        # Фильтруем по пользователю
        user1_logs = get_execution_logs(limit=10, user="user1")
        assert "user1" in user1_logs
        assert "cmd1" in user1_logs
        assert "cmd3" in user1_logs

        # Фильтруем только заблокированные
        blocked_logs = get_execution_logs(limit=10, blocked_only=True)
        # Должен вернуть пустой результат или сообщение
        assert len(blocked_logs) > 0  # Возвращает строку

    def test_log_file_format(self, mock_log_path):
        """Проверка формата JSON Lines"""
        from llm_cli.execution_logger import log_command_execution

        log_command_execution("test cmd", 0, 100, "agent", False)

        # Читаем файл
        content = mock_log_path.read_text()
        lines = content.strip().split("\n")

        # Каждая строка должна быть валидным JSON
        for line in lines:
            if line.strip():
                record = json.loads(line)
                assert "timestamp" in record
                assert "cmd" in record or "type" in record

    def test_clear_logs(self, mock_log_path):
        """Очистка логов"""
        from llm_cli.execution_logger import log_command_execution, clear_execution_logs, get_execution_logs

        # Добавляем логи
        log_command_execution("cmd1", 0, 100, "agent", False)
        log_command_execution("cmd2", 0, 100, "agent", False)

        logs_before = get_execution_logs(limit=10)
        assert "cmd1" in logs_before
        assert "cmd2" in logs_before

        # Очищаем
        clear_execution_logs()

        logs_after = get_execution_logs(limit=10)
        assert "cmd1" not in logs_after
        assert "cmd2" not in logs_after


class TestDatabaseLoggerIntegration:
    """Интеграционные тесты для БД и логгера"""

    @pytest.fixture
    def full_setup(self, tmp_path, monkeypatch):
        """Полная настройка"""
        # Mock для БД
        db_path = tmp_path / "search.db"
        # Mock для логов
        log_path = tmp_path / "execution.log"

        with (
            patch("llm_cli.search_history_db._get_db_path", return_value=db_path),
            patch("llm_cli.execution_logger.LOG_FILE", log_path)
        ):
            yield tmp_path

    def test_full_workflow(self, full_setup):
        """Полный workflow: поиск → логирование → анализ"""
        from llm_cli.search_history_db import init_db, save_search, list_recent_searches
        from llm_cli.execution_logger import log_command_execution, get_execution_logs

        # 1. Инициализируем БД
        conn = init_db()

        # 2. Логируем поиск как команду
        log_command_execution(
            cmd="search 'python tutorials'",
            exit_code=0,
            duration_ms=500,
            user="agent",
            blocked=False
        )

        # 3. Сохраняем результат поиска
        search_results = {
            "results": [
                {"title": "Python Docs", "url": "https://python.org"},
                {"title": "Real Python", "url": "https://realpython.com"}
            ]
        }
        save_search(
            conn,
            session_id="session_1",
            query="python tutorials",
            results=search_results,
            summary="Found Python learning resources"
        )
        conn.close()

        # 4. Проверяем логи
        logs = get_execution_logs(limit=10)
        assert len(logs) > 0
        assert any("search" in log.get("cmd", "").lower() for log in logs)

        # 5. Проверяем историю поиска
        conn = init_db()
        searches = list_recent_searches(conn, session_id="session_1", limit=10)
        conn.close()

        assert len(searches) == 1
        assert searches[0]["query"] == "python tutorials"

    def test_audit_trail(self, full_setup):
        """Аудит-трейл всех операций"""
        from llm_cli.execution_logger import log_command_execution, log_python_execution, get_execution_logs

        # Логируем разные операции
        log_command_execution("git status", 0, 120, "agent", False)
        log_python_execution("hash1", True, 300, "agent")
        log_command_execution("blocked_cmd", None, 0, "agent", True, "security")
        log_python_execution("hash2", False, 50, "agent", "Error")

        # Получаем полный аудит
        all_logs = get_execution_logs(limit=100)

        assert len(all_logs) == 4

        # Анализируем статистику
        commands = [l for l in all_logs if l.get("type") == "command"]
        python = [l for l in all_logs if l.get("type") == "python"]
        blocked = [l for l in all_logs if l.get("blocked")]
        errors = [l for l in all_logs if not l.get("success", True)]

        assert len(commands) == 3
        assert len(python) == 2
        assert len(blocked) == 1
        assert len(errors) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])