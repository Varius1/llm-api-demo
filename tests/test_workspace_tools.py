"""
Комплексные тесты для workspace_tools.py

Демонстрируют все возможности работы с файлами:
- Чтение файлов с разными кодировками
- Запись и редактирование файлов
- Поиск файлов и текста
- Безопасность и защита от атак
- Обработка ошибок и edge cases
"""

import pytest
from pathlib import Path
from unittest.mock import patch
from llm_cli.workspace_tools import (
    read_file,
    write_file,
    edit_file,
    list_directory,
    search_files,
    search_in_files,
    count_lines,
    find_todos,
    detect_language,
    is_path_safe,
    load_gitignore_patterns,
    init_workspace,
)


# Fixture для установки workspace
@pytest.fixture
def workspace_setup(tmp_path):
    """Устанавливает workspace_root для тестов"""
    with patch('llm_cli.workspace_tools._workspace_root', tmp_path):
        yield tmp_path


class TestReadFile:
    """Тесты для функции read_file"""

    def test_read_simple_file(self, workspace_setup):
        """Чтение простого текстового файла"""
        tmp_path = workspace_setup
        
        test_file = tmp_path / "test.txt"
        content = "Строка 1\nСтрока 2\nСтрока 3"
        test_file.write_text(content, encoding="utf-8")

        result = read_file("test.txt")

        assert "1|Строка 1" in result
        assert "2|Строка 2" in result
        assert "3|Строка 3" in result

    def test_read_nonexistent_file(self, workspace_setup):
        """Чтение несуществующего файла"""
        result = read_file("nonexistent.txt")
        assert "Ошибка" in result or "не найден" in result

    def test_read_with_line_range(self, workspace_setup):
        """Чтение диапазона строк"""
        tmp_path = workspace_setup
        
        test_file = tmp_path / "test.txt"
        lines = [f"Строка {i}\n" for i in range(1, 11)]
        test_file.write_text("".join(lines), encoding="utf-8")

        result = read_file("test.txt", start_line=3, end_line=6)

        # Функция возвращает перенумерованные строки, проверяем что строки 3-6 присутствуют
        assert "Строка 3" in result
        assert "Строка 6" in result
        assert "Строка 1" not in result

    def test_read_utf8_encoding(self, workspace_setup):
        """Чтение UTF-8 файла с кириллицей и эмодзи"""
        tmp_path = workspace_setup
        
        test_file = tmp_path / "unicode.txt"
        content = "Привет, мир! 👋\n日本語テスト\n한국어 테스트"
        test_file.write_text(content, encoding="utf-8")

        result = read_file("unicode.txt")
        assert "Привет, мир!" in result
        assert "👋" in result

    def test_read_latin1_fallback(self, workspace_setup):
        """Чтение файла с Latin-1 кодировкой"""
        tmp_path = workspace_setup
        
        test_file = tmp_path / "latin1.txt"
        test_file.write_bytes(b"caf\xe9 r\xe8ve\n")

        result = read_file("latin1.txt")
        assert "caf" in result

    def test_read_large_file_rejected(self, workspace_setup):
        """Файл > 10 MB отклонён"""
        tmp_path = workspace_setup
        
        test_file = tmp_path / "large.txt"
        test_file.write_text("x" * (15 * 1024 * 1024))

        result = read_file("large.txt", max_size_mb=10)
        assert "Ошибка" in result or "большой" in result.lower()

    def test_read_python_file_with_metadata(self, workspace_setup):
        """Чтение Python файла с метаданными"""
        tmp_path = workspace_setup
        
        test_file = tmp_path / "example.py"
        content = """def hello():
    print("Hello, World!")

class MyClass:
    pass
"""
        test_file.write_text(content, encoding="utf-8")

        result = read_file("example.py")
        assert "Size:" in result or "Размер:" in result

    def test_read_path_traversal_protection(self, workspace_setup):
        """Защита от path traversal"""
        result = read_file("../../etc/passwd")
        assert "Ошибка" in result or "пределах" in result


class TestWriteFile:
    """Тесты для функции write_file"""

    def test_write_new_file(self, workspace_setup):
        """Создание нового файла"""
        tmp_path = workspace_setup
        
        content = "Новое содержимое файла\nВторая строка"
        result = write_file("new_file.txt", content)

        assert (tmp_path / "new_file.txt").exists()
        assert (tmp_path / "new_file.txt").read_text() == content

    def test_write_existing_file(self, workspace_setup):
        """Перепись существующего файла"""
        tmp_path = workspace_setup
        
        file_path = tmp_path / "existing.txt"
        file_path.write_text("Старое содержимое")

        result = write_file("existing.txt", "Новое содержимое")
        assert file_path.read_text() == "Новое содержимое"

    def test_write_creates_directories(self, workspace_setup):
        """Автоматическое создание директорий"""
        tmp_path = workspace_setup
        
        result = write_file("subdir/nested/file.txt", "Содержимое")
        assert (tmp_path / "subdir" / "nested" / "file.txt").exists()

    def test_write_with_diff(self, workspace_setup):
        """Генерация diff при изменении"""
        tmp_path = workspace_setup
        
        file_path = tmp_path / "file.txt"
        file_path.write_text("Строка 1\nСтрока 2\nСтрока 3")

        new_content = "Строка 1\nНовая строка 2\nСтрока 3\nСтрока 4"
        result = write_file("file.txt", new_content, show_diff=True)

        assert file_path.read_text() == new_content

    def test_write_allowed_extensions(self, workspace_setup):
        """Запись разрешённых расширений"""
        tmp_path = workspace_setup
        
        extensions = [".py", ".js", ".kt", ".kts", ".md", ".txt", ".json", ".toml", ".yml"]
        
        for ext in extensions:
            result = write_file(f"test{ext}", f"Content for {ext}")
            assert (tmp_path / f"test{ext}").exists()

    def test_write_empty_content(self, workspace_setup):
        """Запись пустого файла"""
        tmp_path = workspace_setup
        
        result = write_file("empty.txt", "")
        assert (tmp_path / "empty.txt").exists()
        assert (tmp_path / "empty.txt").read_text() == ""


class TestEditFile:
    """Тесты для функции edit_file"""

    def test_edit_single_occurrence(self, workspace_setup):
        """Замена одного вхождения"""
        tmp_path = workspace_setup
        
        file_path = tmp_path / "file.txt"
        file_path.write_text("foo bar foo baz")

        result = edit_file("file.txt", "bar", "qux", replace_all=False)
        content = file_path.read_text()
        assert "foo qux foo baz" == content

    def test_edit_all_occurrences(self, workspace_setup):
        """Замена всех вхождений"""
        tmp_path = workspace_setup
        
        file_path = tmp_path / "file.txt"
        file_path.write_text("foo bar foo bar foo")

        result = edit_file("file.txt", "bar", "qux", replace_all=True)
        assert file_path.read_text() == "foo qux foo qux foo"

    def test_edit_not_found(self, workspace_setup):
        """Замена текста которого нет"""
        tmp_path = workspace_setup
        
        file_path = tmp_path / "file.txt"
        file_path.write_text("foo bar baz")

        result = edit_file("file.txt", "notfound", "replacement")
        assert file_path.read_text() == "foo bar baz"

    def test_edit_preserves_formatting(self, workspace_setup):
        """Сохранение форматирования и отступов"""
        tmp_path = workspace_setup
        
        file_path = tmp_path / "code.py"
        content = """def hello():
    print("Hello")
    return True
"""
        file_path.write_text(content)

        result = edit_file("code.py", 'print("Hello")', 'print("World")')
        new_content = file_path.read_text()
        assert '    print("World")' in new_content

    def test_edit_multiline_text(self, workspace_setup):
        """Замена многострочного текста"""
        tmp_path = workspace_setup
        
        file_path = tmp_path / "file.txt"
        content = "line1\nold_line2\nold_line3\nline4"
        file_path.write_text(content)

        result = edit_file("file.txt", "old_line2\nold_line3", "new_line2\nnew_line3")
        assert file_path.read_text() == "line1\nnew_line2\nnew_line3\nline4"


class TestListDirectory:
    """Тесты для функции list_directory"""

    def test_list_empty_directory(self, workspace_setup):
        """Просмотр пустой директории"""
        tmp_path = workspace_setup
        
        result = list_directory(".")
        assert str(tmp_path) in result or "." in result

    def test_list_files_and_directories(self, workspace_setup):
        """Просмотр файлов и директорий"""
        tmp_path = workspace_setup
        
        (tmp_path / "file1.txt").write_text("content")
        (tmp_path / "file2.py").write_text("# Python")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("nested")

        result = list_directory(".", max_depth=2)

        assert "file1.txt" in result
        assert "file2.py" in result
        assert "subdir" in result

    def test_list_max_depth(self, workspace_setup):
        """Ограничение глубины рекурсии"""
        tmp_path = workspace_setup
        
        deep = tmp_path / "level1" / "level2" / "level3"
        deep.mkdir(parents=True)
        (deep / "deep.txt").write_text("deep")

        result = list_directory(".", max_depth=1)

        assert "level1" in result
        # level2 может быть виден как часть дерева, проверяем level3 не виден
        assert "level3" not in result
        assert "deep.txt" not in result


class TestSearchFiles:
    """Тесты для функции search_files"""

    def test_search_glob_pattern(self, workspace_setup):
        """Поиск по glob-паттерну"""
        tmp_path = workspace_setup
        
        (tmp_path / "test1.py").write_text("# Python 1")
        (tmp_path / "test2.py").write_text("# Python 2")
        (tmp_path / "test.js").write_text("// JS")

        result = search_files("*.py", max_results=10)

        assert "test1.py" in result
        assert "test2.py" in result
        assert "test.js" not in result

    def test_search_recursive(self, workspace_setup):
        """Рекурсивный поиск"""
        tmp_path = workspace_setup
        
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.py").write_text("# Nested")
        (tmp_path / "root.py").write_text("# Root")

        result = search_files("**/*.py", max_results=10)

        assert "root.py" in result
        assert "nested.py" in result


class TestSearchInFiles:
    """Тесты для функции search_in_files"""

    def test_search_text_in_files(self, workspace_setup):
        """Поиск текста в файлах"""
        tmp_path = workspace_setup
        
        (tmp_path / "file1.txt").write_text("hello world\nfoo bar")
        (tmp_path / "file2.txt").write_text("goodbye world\nbaz qux")

        result = search_in_files("world", ".", file_pattern="*.txt")

        assert "world" in result

    def test_search_regex_pattern(self, workspace_setup):
        """Поиск по регулярному выражению"""
        tmp_path = workspace_setup
        
        (tmp_path / "log.txt").write_text("ERROR: something failed\nINFO: ok\nERROR: another error")

        result = search_in_files("ERROR.*", ".", file_pattern="*.txt")

        assert "ERROR" in result

    def test_search_case_sensitive(self, workspace_setup):
        """Case-sensitive поиск"""
        tmp_path = workspace_setup
        
        (tmp_path / "file.txt").write_text("Hello HELLO hello")

        result = search_in_files("HELLO", ".", case_sensitive=True)
        assert "HELLO" in result


class TestUtilities:
    """Тесты утилитарных функций"""

    def test_count_lines(self, workspace_setup):
        """Подсчёт строк"""
        tmp_path = workspace_setup
        
        file_path = tmp_path / "code.py"
        content = '''# Comment
def hello():  # Inline comment
    """Hello"""  # Docstring
    pass

# Another comment
class Test:  # Comment
    """Class docstring"""
    def method(self):
        # Inline
        return True
'''
        file_path.write_text(content)

        result = count_lines("code.py")
        assert "Всего строк:" in result or "Код:" in result

    def test_find_todos(self, workspace_setup):
        """Поиск TODO, FIXME и других маркеров"""
        tmp_path = workspace_setup
        
        file_path = tmp_path / "code.py"
        content = '''def function():
    # TODO: implement this
    x = 1  # FIXME: bug here
    # HACK: temporary solution
    return x  # NOTE: important
'''
        file_path.write_text(content)

        result = find_todos(".")

        assert "TODO" in result
        assert "FIXME" in result
        assert "HACK" in result

    def test_detect_language(self, workspace_setup):
        """Определение языка по расширению"""
        tmp_path = workspace_setup
        
        (tmp_path / "test.py").write_text("# Python")
        assert detect_language("test.py") == "python"

        (tmp_path / "script.js").write_text("// JS")
        assert detect_language("script.js") == "javascript"

        (tmp_path / "app.ts").write_text("// TS")
        assert detect_language("app.ts") == "typescript"


class TestSecurity:
    """Тесты безопасности"""

    def test_is_path_safe_inside_workspace(self, workspace_setup):
        """Безопасный путь внутри workspace"""
        tmp_path = workspace_setup
        
        file_path = tmp_path / "safe" / "file.txt"
        file_path.parent.mkdir()
        file_path.write_text("safe")

        assert is_path_safe("safe/file.txt")

    def test_is_path_safe_outside_workspace(self, workspace_setup):
        """Путь вне workspace небезопасен"""
        assert not is_path_safe("/etc/passwd")

    def test_is_path_safe_traversal(self, workspace_setup):
        """Path traversal атака заблокирована"""
        assert not is_path_safe("../../etc/passwd")

    def test_gitignore_patterns_loading(self, workspace_setup):
        """Загрузка паттернов из .gitignore"""
        tmp_path = workspace_setup
        
        (tmp_path / ".gitignore").write_text("*.log\n__pycache__/\n*.pyc\n")

        patterns = load_gitignore_patterns(tmp_path)

        assert "*.log" in patterns
        assert "__pycache__/" in patterns


class TestIntegration:
    """Интеграционные тесты"""

    def test_read_edit_write_cycle(self, workspace_setup):
        """Полный цикл: чтение → редактирование → запись"""
        tmp_path = workspace_setup
        
        # 1. Запись
        write_file("document.txt", "Hello World\nThis is a test")

        # 2. Чтение
        read_result = read_file("document.txt")
        assert "Hello World" in read_result

        # 3. Редактирование
        edit_file("document.txt", "Hello", "Goodbye", replace_all=False)

        # 4. Проверка
        final_read = read_file("document.txt")
        assert "Goodbye World" in final_read
        assert "Hello" not in final_read

    def test_workflow_create_list_edit_search(self, workspace_setup):
        """Реалистичный workflow проекта"""
        tmp_path = workspace_setup
        
        # 1. Создаём структуру
        src_dir = tmp_path / "src"
        src_dir.mkdir()

        write_file("src/main.py", 'def main():\n    print("Hello")\n')
        write_file("src/utils.py", 'def helper():\n    # TODO: optimize\n    return 42\n')

        # 2. Просматриваем
        dir_result = list_directory(".", max_depth=2)
        assert "src" in dir_result

        # 3. Ищем файлы
        search_result = search_files("**/*.py", max_results=10)
        assert "main.py" in search_result or "src/main.py" in search_result

        # 4. Ищем TODO
        todo_result = find_todos(".")
        assert "TODO" in todo_result

        # 5. Редактируем
        edit_file("src/utils.py", "return 42", "return 100", replace_all=False)

        # 6. Проверяем
        final_content = read_file("src/utils.py")
        assert "return 100" in final_content


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])