"""
Тесты для config.py и memory.py

Демонстрируют работу с:
- Конфигурационными файлами (TOML)
- 3-слойной моделью памяти (JSON)
- Профилями пользователей
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch
import json


class TestConfig:
    """Тесты для config.py"""

    @pytest.fixture
    def mock_config_dir(self, tmp_path, monkeypatch):
        """Mock для директории конфигурации"""
        config_dir = tmp_path / ".config" / "llm-cli"
        config_dir.mkdir(parents=True)
        
        # Переопределяем путь к конфиге
        with patch("llm_cli.config._config_path", return_value=config_dir / "config.toml"):
            yield config_dir

    def test_save_and_load_config(self, mock_config_dir):
        """Сохранение и загрузка конфигурации"""
        from llm_cli.config import AppConfig

        # Создаём и сохраняем конфиг
        config = AppConfig(
            api_key="test-key-123",
            default_model="google/gemma-2-9b-it",
            temperature=0.7,
            workspace_root="/path/to/workspace"
        )
        config.save()

        # Загружаем и проверяем
        loaded = AppConfig.load()

        assert loaded.api_key == "test-key-123"
        assert loaded.default_model == "google/gemma-2-9b-it"
        assert loaded.temperature == 0.7
        assert loaded.workspace_root == "/path/to/workspace"

    def test_load_nonexistent_config(self, mock_config_dir):
        """Загрузка несуществующего конфига"""
        from llm_cli.config import AppConfig

        config = AppConfig.load()

        # Должен вернуть конфиг с дефолтными значениями
        assert config is not None

    def test_load_corrupted_config(self, mock_config_dir):
        """Загрузка повреждённого конфига"""
        config_file = mock_config_dir / "config.toml"
        config_file.write_text("invalid toml content {{{")

        from llm_cli.config import AppConfig

        # Должен обработать ошибку и вернуть дефолтный конфиг
        config = AppConfig.load()
        assert config is not None

    def test_api_key_from_env(self, mock_config_dir, monkeypatch):
        """API ключ из переменной окружения"""
        monkeypatch.setenv("OPENROUTER_API_KEY", "env-api-key-456")

        from llm_cli.config import AppConfig

        config = AppConfig.load()

        # Должен использовать ключ из env
        assert config.api_key == "env-api-key-456"

    def test_config_file_format(self, mock_config_dir):
        """Проверка формата TOML файла"""
        from llm_cli.config import AppConfig

        config = AppConfig(
            api_key="test-key",
            default_model="test-model",
            temperature=0.5
        )
        config.save()

        config_file = mock_config_dir / "config.toml"
        content = config_file.read_text()

        # Проверяем структуру TOML
        assert "[general]" in content
        assert "api_key = " in content
        assert "default_model = " in content


class TestMemory:
    """Тесты для memory.py"""

    @pytest.fixture
    def mock_memory_paths(self, tmp_path, monkeypatch):
        """Mock для путей памяти"""
        config_dir = tmp_path / ".config" / "llm-cli"
        config_dir.mkdir(parents=True)

        with (
            patch("llm_cli.memory._long_term_path", return_value=config_dir / "long_term_memory.json"),
            patch("llm_cli.memory._history_path", return_value=config_dir / "history.json")
        ):
            yield config_dir

    def test_working_memory_operations(self, mock_memory_paths):
        """Операции с рабочей памятью"""
        from llm_cli.memory import WorkingMemory, MemoryManager

        memory = MemoryManager()

        # Создаём рабочую память
        working = WorkingMemory(
            task="Test task",
            goals=["Goal 1", "Goal 2"],
            facts={"key1": "value1"},
            notes=["Note 1"]
        )
        memory.working_memory = working

        # Проверяем, что данные сохранены
        assert memory.working_memory.task == "Test task"
        assert len(memory.working_memory.goals) == 2
        assert memory.working_memory.facts["key1"] == "value1"

    def test_long_term_memory_persistence(self, mock_memory_paths):
        """Персистентность долговременной памяти"""
        from llm_cli.memory import MemoryManager

        # Создаём менеджер и сохраняем знания
        memory1 = MemoryManager()
        memory1.remember_knowledge("test_key", "test_value")
        memory1.remember_decision("Test decision")

        # Создаём новый менеджер (должен загрузить из файла)
        memory2 = MemoryManager()

        assert memory2.long_term_memory.knowledge.get("test_key") == "test_value"
        assert len(memory2.long_term_memory.decisions) > 0

    def test_profile_operations(self, mock_memory_paths):
        """Операции с профилями пользователей"""
        from llm_cli.memory import UserProfile, MemoryManager

        memory = MemoryManager()

        # Создаём профиль
        profile = UserProfile(
            name="Test User",
            language="Russian",
            style="Professional",
            format="Detailed",
            expertise=["Python", "AI"],
            domain="Software Development"
        )

        memory.save_profile("test_profile", profile)

        # Загружаем профиль
        loaded_profile = memory.get_profile("test_profile")

        assert loaded_profile.name == "Test User"
        assert loaded_profile.language == "Russian"
        assert "Python" in loaded_profile.expertise

    def test_context_block_generation(self, mock_memory_paths):
        """Генерация контекстного блока"""
        from llm_cli.memory import MemoryManager

        memory = MemoryManager()

        # Заполняем память
        memory.remember_knowledge("python", "Python is a programming language")
        memory.remember_knowledge("ai", "AI is artificial intelligence")
        memory.remember_decision("Use Python for the project")

        # Генерируем контекстный блок
        context_block = memory.get_context_block()

        # Проверяем, что знания включены
        assert "python" in context_block.lower() or "Python" in context_block
        assert "knowledge" in context_block.lower() or "Знания" in context_block

    def test_forget_operation(self, mock_memory_paths):
        """Удаление ключа из памяти"""
        from llm_cli.memory import MemoryManager

        memory = MemoryManager()

        # Добавляем и удаляем
        memory.remember_knowledge("to_forget", "value")
        memory.forget("to_forget")

        assert "to_forget" not in memory.long_term_memory.knowledge

    def test_multiple_profiles(self, mock_memory_paths):
        """Работа с несколькими профилями"""
        from llm_cli.memory import UserProfile, MemoryManager

        memory = MemoryManager()

        # Создаём несколько профилей
        profile1 = UserProfile(name="Developer", language="English", style="Technical")
        profile2 = UserProfile(name="Designer", language="Russian", style="Creative")

        memory.save_profile("dev", profile1)
        memory.save_profile("design", profile2)

        # Проверяем оба профиля
        assert memory.get_profile("dev").name == "Developer"
        assert memory.get_profile("design").name == "Designer"

    def test_memory_file_format(self, mock_memory_paths):
        """Проверка формата JSON файла"""
        from llm_cli.memory import MemoryManager

        memory = MemoryManager()
        memory.remember_knowledge("test", "value")

        # Проверяем файл
        ltm_file = mock_memory_paths / "long_term_memory.json"
        assert ltm_file.exists()

        content = json.loads(ltm_file.read_text())
        assert "knowledge" in content
        assert content["knowledge"]["test"] == "value"


class TestMemoryIntegration:
    """Интеграционные тесты для системы памяти"""

    @pytest.fixture
    def full_memory_setup(self, tmp_path, monkeypatch):
        """Полная настройка системы памяти"""
        config_dir = tmp_path / ".config" / "llm-cli"
        config_dir.mkdir(parents=True)

        with (
            patch("llm_cli.memory._long_term_path", return_value=config_dir / "long_term_memory.json"),
            patch("llm_cli.memory._history_path", return_value=config_dir / "history.json")
        ):
            yield config_dir

    def test_full_workflow(self, full_memory_setup):
        """Полный workflow работы с памятью"""
        from llm_cli.memory import UserProfile, MemoryManager

        # 1. Создаём профиль пользователя
        memory = MemoryManager()
        profile = UserProfile(
            name="AI Developer",
            language="Russian",
            style="Concise",
            expertise=["Python", "Machine Learning"]
        )
        memory.save_profile("default", profile)

        # 2. Сохраняем знания в процессе работы
        memory.remember_knowledge("project_stack", "Python, FastAPI, PostgreSQL")
        memory.remember_knowledge("coding_style", "PEP 8, type hints")
        memory.remember_decision("Use async/await for I/O operations")

        # 3. Создаём новый экземпляр (симуляция нового запуска)
        memory2 = MemoryManager()

        # 4. Проверяем, что всё сохранилось
        assert memory2.get_profile("default").name == "AI Developer"
        assert memory2.long_term_memory.knowledge["project_stack"] == "Python, FastAPI, PostgreSQL"
        assert len(memory2.long_term_memory.decisions) > 0

        # 5. Обновляем знания
        memory2.remember_knowledge("new_feature", "Added MCP support")

        # 6. Генерируем контекст для следующей сессии
        context = memory2.get_context_block()
        assert len(context) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])