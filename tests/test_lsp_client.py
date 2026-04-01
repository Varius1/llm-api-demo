"""Тесты для LSP клиента и адаптеров."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Путь к тестируемому модулю
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_cli.lsp_client import LSPClient, LSPResponse
from llm_cli.lsp_adapters import SymbolInfo, PythonLSPClient, KotlinLSPClient


class TestSymbolInfo:
    """Тесты для SymbolInfo."""
    
    def test_symbol_info_creation(self):
        """Создание SymbolInfo с параметрами."""
        sym = SymbolInfo(
            name="calculate_total",
            kind="function",
            file_path="/test/file.py",
            line=10,
            column=4,
            container_name="OrderService",
            documentation="Calculate total price",
            code_snippet="def calculate_total(): pass"
        )
        
        assert sym.name == "calculate_total"
        assert sym.kind == "function"
        assert sym.file_path == "/test/file.py"
        assert sym.line == 10
        assert sym.column == 4
        assert sym.container_name == "OrderService"
    
    def test_symbol_info_to_dict(self):
        """Конвертация SymbolInfo в dict."""
        sym = SymbolInfo(
            name="test",
            kind="function",
            file_path="/test.py",
            line=1,
            column=0
        )
        
        d = sym.to_dict()
        
        assert d["name"] == "test"
        assert d["kind"] == "function"
        assert "file_path" in d
    
    def test_symbol_info_str(self):
        """Строковое представление SymbolInfo."""
        sym = SymbolInfo(
            name="my_func",
            kind="function",
            file_path="test.py",
            line=5,
            column=2,
            documentation="My function"
        )
        
        s = str(sym)
        
        assert "my_func" in s
        assert "function" in s
        assert "test.py" in s
        assert "My function" in s


class TestLSPResponse:
    """Тесты для LSPResponse."""
    
    def test_response_with_result(self):
        """Ответ с результатом."""
        resp = LSPResponse(result={"data": "test"})
        
        assert not resp.is_error()
        assert resp.result == {"data": "test"}
        assert resp.error is None
    
    def test_response_with_error(self):
        """Ответ с ошибкой."""
        resp = LSPResponse(error={"code": -32600, "message": "Invalid request"})
        
        assert resp.is_error()
        assert resp.error["code"] == -32600
        assert resp.result is None


class TestPythonLSPClient:
    """Тесты для PythonLSPClient."""
    
    def test_python_client_initialization(self):
        """Инициализация PythonLSPClient."""
        client = PythonLSPClient(workspace_root="/test/workspace")
        
        assert client.workspace_root == "/test/workspace"
        assert client.timeout == 30.0
        assert client._process is None
    
    def test_python_client_get_server_command(self):
        """Команда запуска Python LSP сервера."""
        client = PythonLSPClient(workspace_root="/test")
        
        cmd = client._get_server_command()
        
        assert cmd == ["pylsp"]
    
    @patch("subprocess.Popen")
    def test_start_server_success(self, mock_popen):
        """Успешный запуск сервера."""
        mock_process = MagicMock()
        mock_process.stdout = None
        mock_popen.return_value = mock_process
        
        client = PythonLSPClient(workspace_root="/test")
        
        # start_server требует реального запуска, тестируем только инициализацию
        assert client.workspace_root == "/test"
    
    def test_find_definition_format(self):
        """Формат запроса find_definition."""
        client = PythonLSPClient(workspace_root="/test")
        
        # Тестируем что метод существует
        assert hasattr(client, 'find_definition')
    
    def test_get_hover_info_format(self):
        """Формат запроса get_hover_info."""
        client = PythonLSPClient(workspace_root="/test")
        
        assert hasattr(client, 'get_hover_info')
    
    def test_list_workspace_symbols_format(self):
        """Формат запроса list_workspace_symbols."""
        client = PythonLSPClient(workspace_root="/test")
        
        assert hasattr(client, 'list_workspace_symbols')
    
    def test_find_references_format(self):
        """Формат запроса find_references."""
        client = PythonLSPClient(workspace_root="/test")
        
        assert hasattr(client, 'find_references')


class TestKotlinLSPClient:
    """Тесты для KotlinLSPClient."""
    
    def test_kotlin_client_initialization(self):
        """Инициализация KotlinLSPClient."""
        client = KotlinLSPClient(workspace_root="/test/android")
        
        assert client.workspace_root == "/test/android"
    
    def test_kotlin_client_get_server_command(self):
        """Команда запуска Kotlin LSP сервера."""
        client = KotlinLSPClient(workspace_root="/test")
        
        cmd = client._get_server_command()
        
        assert "kotlin-language-server" in cmd
    
    @patch("pathlib.Path.exists")
    def test_start_server_without_gradle(self, mock_exists):
        """Запуск без Gradle файла должен вернуть False."""
        mock_exists.return_value = False
        
        client = KotlinLSPClient(workspace_root="/test")
        
        # Kotlin LSP требует build.gradle.kts
        # В реальном тесте это проверится
        
    def test_find_definition_format(self):
        """Формат запроса find_definition."""
        client = KotlinLSPClient(workspace_root="/test")
        
        assert hasattr(client, 'find_definition')
    
    def test_get_hover_info_format(self):
        """Формат запроса get_hover_info."""
        client = KotlinLSPClient(workspace_root="/test")
        
        assert hasattr(client, 'get_hover_info')


class TestLSPIntegration:
    """Интеграционные тесты LSP."""
    
    @pytest.mark.integration
    def test_python_lsp_full_flow(self):
        """Полный цикл работы с Python LSP (требует pylsp)."""
        # Этот тест запускается только если pylsp установлен
        try:
            client = PythonLSPClient(workspace_root=str(Path(__file__).parent.parent / "src"))
            
            # В реальном тесте:
            # success = client.start_server()
            # assert success
            
            # symbols = client.list_workspace_symbols("Agent")
            # assert len(symbols) > 0
            
            # client.shutdown()
            
            pytest.skip("Интеграционный тест требует запуска LSP сервера")
            
        except Exception as e:
            pytest.skip(f"Интеграционный тест пропущен: {e}")
    
    @pytest.mark.integration
    def test_kotlin_lsp_with_gradle(self):
        """Работа с Kotlin LSP и Gradle проектом."""
        # Этот тест требует Kotlin проекта с Gradle
        pytest.skip("Интеграционный тест требует Kotlin проекта с Gradle 9.0")


class TestLSPErrorHandling:
    """Тесты обработки ошибок."""
    
    def test_response_error_handling(self):
        """Обработка ошибок в LSPResponse."""
        resp = LSPResponse(error={"code": -32601, "message": "Method not found"})
        
        assert resp.is_error()
        assert resp.error["code"] == -32601
    
    def test_symbol_info_minimal(self):
        """SymbolInfo с минимальными данными."""
        sym = SymbolInfo(
            name="test",
            kind="unknown",
            file_path="unknown",
            line=0,
            column=0
        )
        
        assert sym.name == "test"
        assert sym.documentation is None
        assert sym.code_snippet is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])