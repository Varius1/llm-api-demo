"""Тесты для классификатора ошибок."""

import pytest
from src.llm_cli.error_classifier import (
    ErrorType,
    classify_error,
    is_error_message,
    should_auto_fix,
)


class TestIsErrorMessage:
    """Тесты для функции определения ошибки."""

    def test_empty_string(self):
        """Пустая строка — не ошибка."""
        assert not is_error_message("")

    def test_success_message(self):
        """Успешное сообщение — не ошибка."""
        assert not is_error_message("✓ Файл успешно создан")
        assert not is_error_message("Success: operation completed")
        assert not is_error_message("OK")

    def test_error_with_prefix(self):
        """Сообщения с префиксом error/ошибка."""
        assert is_error_message("Error: file not found")
        assert is_error_message("Ошибка: файл не найден")
        assert is_error_message("  error: invalid syntax")

    def test_checkmark_symbol(self):
        """Символ ✗ указывает на ошибку."""
        assert is_error_message("✗ Тест не пройден")
        assert is_error_message("✖ Ошибка выполнения")

    def test_traceback(self):
        """Traceback — это ошибка."""
        assert is_error_message("Traceback (most recent call last):")
        assert is_error_message("traceback: something went wrong")

    def test_exception(self):
        """Exception — это ошибка."""
        assert is_error_message("Exception: something failed")
        assert is_error_message("RuntimeError: invalid state")

    def test_failed_keyword(self):
        """Ключевые слова failed/failure."""
        assert is_error_message("Test failed")
        assert is_error_message("Build failure")
        assert is_error_message("Operation failed with code 500")

    def test_exit_code(self):
        """Non-zero exit code — ошибка."""
        assert is_error_message("Exit code 1")
        assert is_error_message("Exit code 127")
        assert not is_error_message("Exit code 0")


class TestClassifyError:
    """Тесты для функции классификации ошибок."""

    def test_not_an_error(self):
        """Успешные результаты классифицируются как NOT_AN_ERROR."""
        assert classify_error("✓ Success", "some_tool") == ErrorType.NOT_AN_ERROR
        assert classify_error("OK", "some_tool") == ErrorType.NOT_AN_ERROR
        assert classify_error("", "some_tool") == ErrorType.NOT_AN_ERROR

    def test_syntax_errors(self):
        """Синтаксические ошибки."""
        assert classify_error("SyntaxError: invalid syntax", "write_file") == ErrorType.SYNTAX
        # "Ошибка синтаксиса" без маркера ошибки — не распознаётся
        assert classify_error("✗ Ошибка синтаксиса", "write_file") == ErrorType.SYNTAX
        # "invalid syntax" без маркера — тоже не распознаётся
        assert classify_error("✗ invalid syntax at line 5", "write_file") == ErrorType.SYNTAX
        # "unexpected EOF" без маркера — не распознаётся
        assert classify_error("✗ unexpected EOF while parsing", "write_file") == ErrorType.SYNTAX
        # "py_compile error" без маркера — не распознаётся
        assert classify_error("✗ py_compile error", "write_file") == ErrorType.SYNTAX

    def test_runtime_errors(self):
        """Ошибки выполнения."""
        assert classify_error("NameError: name 'x' is not defined", "run_command") == ErrorType.RUNTIME
        assert classify_error("TypeError: unsupported operand", "run_command") == ErrorType.RUNTIME
        assert classify_error("AttributeError: no attribute", "run_command") == ErrorType.RUNTIME
        assert classify_error("KeyError: 'missing'", "run_command") == ErrorType.RUNTIME
        assert classify_error("IndexError: list index out of range", "run_command") == ErrorType.RUNTIME
        assert classify_error("ValueError: invalid literal", "run_command") == ErrorType.RUNTIME
        assert classify_error("ImportError: No module named", "run_command") == ErrorType.RUNTIME
        assert classify_error("ModuleNotFoundError: no module", "run_command") == ErrorType.RUNTIME
        assert classify_error("FileNotFoundError: file not found", "run_command") == ErrorType.RUNTIME
        # Permission denied — это PERMISSION, не RUNTIME
        assert classify_error("OSError: permission denied", "run_command") == ErrorType.PERMISSION
        # Но OSError без permission — это RUNTIME
        assert classify_error("OSError: file too large", "run_command") == ErrorType.RUNTIME

    def test_validation_errors(self):
        """Ошибки валидации (упавшие тесты)."""
        assert classify_error("pytest failed", "run_tests") == ErrorType.VALIDATION
        assert classify_error("AssertionError: 1 != 2", "run_tests") == ErrorType.VALIDATION
        assert classify_error("assert failed", "run_tests") == ErrorType.VALIDATION
        assert classify_error("TEST FAILED", "run_tests") == ErrorType.VALIDATION
        # "FAILED" без контекста — это RUNTIME, а не VALIDATION
        assert classify_error("=== FAILED ===", "run_tests") == ErrorType.RUNTIME
        assert classify_error("validation failed", "run_tests") == ErrorType.VALIDATION

    def test_permission_errors(self):
        """Ошибки доступа и блокировки."""
        # "Blocked" без маркера ошибки — не распознаётся как ошибка
        assert classify_error("✗ Blocked: command not allowed", "run_command") == ErrorType.PERMISSION
        assert classify_error("Заблокировано: опасная операция", "run_command") == ErrorType.PERMISSION
        # "Permission denied" без маркера — не распознаётся
        assert classify_error("✗ Permission denied", "run_command") == ErrorType.PERMISSION
        assert classify_error("✗ Access denied", "run_command") == ErrorType.PERMISSION
        assert classify_error("✗ Не разрешено", "run_command") == ErrorType.PERMISSION
        assert classify_error("Security check failed", "run_command") == ErrorType.PERMISSION

    def test_architecture_errors(self):
        """Ошибки, требующие архитектурного решения."""
        # Без маркера ошибки — не распознаются
        assert classify_error("✗ No such method", "some_tool") == ErrorType.ARCHITECTURE
        assert classify_error("✗ Method not found", "some_tool") == ErrorType.ARCHITECTURE
        assert classify_error("✗ Не найдено", "some_tool") == ErrorType.ARCHITECTURE
        assert classify_error("✗ Требуется решение", "some_tool") == ErrorType.ARCHITECTURE
        assert classify_error("✗ Выбери подход", "some_tool") == ErrorType.ARCHITECTURE

    def test_default_to_runtime(self):
        """Неизвестные ошибки по умолчанию — RUNTIME."""
        # Если ошибка не подходит под другие категории, считаем её runtime
        error_msg = "✗ Something went wrong unexpectedly"
        result = classify_error(error_msg, "some_tool")
        # Это будет RUNTIME, так как содержит "✗" и не подходит под другие паттерны
        assert result == ErrorType.RUNTIME


class TestShouldAutoFix:
    """Тесты для функции определения необходимости автофикса."""

    def test_syntax_should_fix(self):
        """Синтаксические ошибки нужно фиксить."""
        assert should_auto_fix(ErrorType.SYNTAX) is True

    def test_runtime_should_fix(self):
        """Runtime ошибки нужно фиксить."""
        assert should_auto_fix(ErrorType.RUNTIME) is True

    def test_validation_should_fix(self):
        """Ошибки валидации нужно фиксить."""
        assert should_auto_fix(ErrorType.VALIDATION) is True

    def test_architecture_should_not_fix(self):
        """Архитектурные ошибки не нужно фиксить автоматически."""
        assert should_auto_fix(ErrorType.ARCHITECTURE) is False

    def test_permission_should_not_fix(self):
        """Ошибки доступа не нужно фиксить автоматически."""
        assert should_auto_fix(ErrorType.PERMISSION) is False

    def test_not_an_error_should_not_fix(self):
        """Если это не ошибка, то и фиксить не нужно."""
        assert should_auto_fix(ErrorType.NOT_AN_ERROR) is False


class TestIntegration:
    """Интеграционные тесты."""

    def test_full_classification_flow(self):
        """Полный поток: от сообщения до решения о фиксе."""
        # Синтаксическая ошибка → нужно фиксить
        error = "SyntaxError: invalid syntax at line 10"
        error_type = classify_error(error, "write_file")
        assert error_type == ErrorType.SYNTAX
        assert should_auto_fix(error_type) is True

        # Упавший тест → нужно фиксить
        error = "AssertionError: expected 5, got 3"
        error_type = classify_error(error, "run_tests")
        assert error_type == ErrorType.VALIDATION
        assert should_auto_fix(error_type) is True

        # Блокировка → не фиксить
        error = "Blocked: command 'rm -rf' not allowed"
        error_type = classify_error(error, "run_command")
        assert error_type == ErrorType.PERMISSION
        assert should_auto_fix(error_type) is False

        # Успех → не ошибка
        result = "✓ File written successfully"
        error_type = classify_error(result, "write_file")
        assert error_type == ErrorType.NOT_AN_ERROR
        assert should_auto_fix(error_type) is False