#!/usr/bin/env bash
# Демо-скрипт для записи видео: File Assistant — ассистент для работы с файлами проекта
#
# Запуск: bash scripts/demo_file_assistant.sh
#
# Три сценария (последовательно):
#   1. Поиск использования компонента по всему проекту
#   2. Проверка инвариантов: импорты httpx вне api.py
#   3. Генерация CHANGELOG.md на основе git-истории
#
# Переменные среды:
#   SCENARIO  — запустить только один сценарий: 1, 2 или 3
#   PAUSE     — секунд паузы между сценариями (default: 3)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -f "$PROJECT_ROOT/.venv/bin/python" ]]; then
    PYTHON="$PROJECT_ROOT/.venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    PYTHON="python"
fi

BOLD="\033[1m"
CYAN="\033[1;36m"
GREEN="\033[1;32m"
YELLOW="\033[1;33m"
MAGENTA="\033[1;35m"
DIM="\033[2m"
RESET="\033[0m"

PAUSE="${PAUSE:-3}"

divider() {
    echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n"
}

header() {
    divider
    echo -e "${BOLD}${GREEN}$1${RESET}"
    divider
}

scenario_header() {
    local num="$1"
    local title="$2"
    echo ""
    echo -e "${BOLD}${MAGENTA}╔══════════════════════════════════════════════════╗${RESET}"
    echo -e "${BOLD}${MAGENTA}║  Сценарий $num: $title${RESET}"
    echo -e "${BOLD}${MAGENTA}╚══════════════════════════════════════════════════╝${RESET}"
    echo ""
}

# Читаем конфиг Python-пакета (use_local, local_url, default_model)
read_config() {
    "$PYTHON" -c "
import json
from llm_cli.config import AppConfig
from llm_cli.models import LOCAL_MODEL_ID, OPENROUTER_URL
cfg = AppConfig.load()
print(json.dumps({
    'use_local': cfg.use_local,
    'local_url': cfg.local_url,
    'model': LOCAL_MODEL_ID if cfg.use_local else cfg.default_model,
    'mode': 'local' if cfg.use_local else 'openrouter',
    'api_key_set': bool(cfg.api_key),
}))
" 2>/dev/null
}

run_goal() {
    local goal="$1"
    echo -e "${YELLOW}Задача:${RESET} ${BOLD}${goal}${RESET}"
    echo ""
    echo -e "${DIM}$ llm-cli --file-assistant --file-goal \"${goal}\"${RESET}"
    echo ""
    cd "$PROJECT_ROOT"
    "$PYTHON" -m llm_cli --file-assistant --file-goal "$goal"
}

# ─────────────────────────────────────────────────────────────────────────────
# Заголовок + определение режима
# ─────────────────────────────────────────────────────────────────────────────
clear
header "File Assistant — ассистент для работы с файлами проекта"

# Читаем конфиг чтобы показать реальный режим
CFG_JSON=$(read_config)
if [[ -n "$CFG_JSON" ]]; then
    USE_LOCAL=$(echo "$CFG_JSON" | "$PYTHON" -c "import sys,json; d=json.load(sys.stdin); print(d['use_local'])")
    MODEL=$(echo "$CFG_JSON" | "$PYTHON" -c "import sys,json; d=json.load(sys.stdin); print(d['model'])")
    MODE=$(echo "$CFG_JSON" | "$PYTHON" -c "import sys,json; d=json.load(sys.stdin); print(d['mode'])")
    LOCAL_URL=$(echo "$CFG_JSON" | "$PYTHON" -c "import sys,json; d=json.load(sys.stdin); print(d['local_url'])")
else
    USE_LOCAL="False"
    MODEL="openai/gpt-4o-mini"
    MODE="openrouter"
fi

echo -e "${BOLD}Что будет показано:${RESET}"
echo ""
echo -e "  ${CYAN}Сценарий 1.${RESET} ${BOLD}Поиск компонента${RESET}"
echo -e "             Ассистент находит все места использования MCPSession в коде"
echo ""
echo -e "  ${CYAN}Сценарий 2.${RESET} ${BOLD}Проверка инвариантов${RESET}"
echo -e "             Поиск прямых импортов httpx вне api.py (нарушение правил)"
echo ""
echo -e "  ${CYAN}Сценарий 3.${RESET} ${BOLD}Генерация документации${RESET}"
echo -e "             Создание CHANGELOG.md на основе git-истории и README"
echo ""

if [[ "$USE_LOCAL" == "True" ]]; then
    echo -e "${DIM}Режим:      локальная модель (llama-server)${RESET}"
    echo -e "${DIM}URL:        $LOCAL_URL${RESET}"
    echo -e "${DIM}Модель:     $MODEL${RESET}"
    echo -e "${DIM}MCP-сервер: mcp_server_files (stdio)${RESET}"
    echo ""
    echo -e "${GREEN}Используется локальная модель. Убедитесь что llama-server запущен на $LOCAL_URL${RESET}"
else
    echo -e "${DIM}Режим:      OpenRouter${RESET}"
    echo -e "${DIM}Модель:     $MODEL${RESET}"
    echo -e "${DIM}MCP-сервер: mcp_server_files (stdio)${RESET}"
fi

echo ""
sleep "$PAUSE"

# ─────────────────────────────────────────────────────────────────────────────
# Сценарий 1: Поиск использования компонента
# ─────────────────────────────────────────────────────────────────────────────
if [[ -z "$SCENARIO" || "$SCENARIO" == "1" ]]; then
    scenario_header "1" "Поиск использования MCPSession"

    echo -e "${BOLD}Задача:${RESET} найти все места в Python-коде проекта, где используется класс MCPSession"
    echo ""
    echo -e "${DIM}Инструменты, которые вызовет ассистент:${RESET}"
    echo -e "  ${DIM}get_project_structure${RESET} → изучит структуру проекта"
    echo -e "  ${DIM}grep_in_files${RESET}          → найдёт все вхождения MCPSession"
    echo -e "  ${DIM}read_file${RESET}              → прочитает ключевые места для анализа"
    echo ""
    sleep 2

    run_goal "Найди все места в проекте, где используется класс MCPSession. Покажи файлы, строки и краткий анализ — где создаётся, где импортируется, где передаётся как параметр."

    echo ""
    echo -e "${GREEN}✓ Сценарий 1 завершён${RESET}"
    sleep "$PAUSE"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Сценарий 2: Проверка инвариантов
# ─────────────────────────────────────────────────────────────────────────────
if [[ -z "$SCENARIO" || "$SCENARIO" == "2" ]]; then
    scenario_header "2" "Проверка инвариантов: импорты httpx"

    echo -e "${BOLD}Задача:${RESET} проверить, что прямые импорты httpx встречаются только в разрешённых файлах"
    echo ""
    echo -e "${DIM}Инварианты проекта:${RESET}"
    echo -e "  ${DIM}• httpx импортируется только в src/llm_cli/api.py${RESET}"
    echo -e "  ${DIM}• исключение: mcp_server.py (для get_crypto_price)${RESET}"
    echo ""
    echo -e "${DIM}Инструменты, которые вызовет ассистент:${RESET}"
    echo -e "  ${DIM}grep_in_files${RESET} → найдёт все import httpx"
    echo -e "  ${DIM}read_file${RESET}     → прочитает нарушающие файлы для анализа"
    echo ""
    sleep 2

    run_goal "Проверь инвариант проекта: прямой импорт httpx (import httpx, from httpx) должен встречаться только в src/llm_cli/api.py и src/llm_cli/mcp_server.py. Найди все нарушения в остальных Python-файлах. Сформируй отчёт: нарушения или OK."

    echo ""
    echo -e "${GREEN}✓ Сценарий 2 завершён${RESET}"
    sleep "$PAUSE"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Сценарий 3: Генерация CHANGELOG.md
# ─────────────────────────────────────────────────────────────────────────────
if [[ -z "$SCENARIO" || "$SCENARIO" == "3" ]]; then
    scenario_header "3" "Генерация CHANGELOG.md"

    echo -e "${BOLD}Задача:${RESET} сгенерировать CHANGELOG.md на основе git-истории и структуры проекта"
    echo ""
    echo -e "${DIM}Инструменты, которые вызовет ассистент:${RESET}"
    echo -e "  ${DIM}get_git_log${RESET}              → получит историю коммитов"
    echo -e "  ${DIM}read_file(README.md)${RESET}     → прочитает описание проекта"
    echo -e "  ${DIM}get_project_structure${RESET}    → изучит текущую структуру"
    echo -e "  ${DIM}get_file_diff${RESET}            → покажет diff перед записью"
    echo -e "  ${DIM}write_file(CHANGELOG.md)${RESET} → сохранит результат"
    echo ""
    sleep 2

    run_goal "Сгенерируй CHANGELOG.md для проекта llm-api-demo. Для этого: прочитай README.md для понимания проекта, получи последние 15 коммитов git log, изучи структуру проекта. Составь CHANGELOG в формате Keep a Changelog с секциями Added/Changed/Fixed на основе коммитов. Покажи diff и сохрани файл CHANGELOG.md в корень проекта."

    echo ""
    echo -e "${GREEN}✓ Сценарий 3 завершён${RESET}"

    if [[ -f "$PROJECT_ROOT/CHANGELOG.md" ]]; then
        echo ""
        echo -e "${YELLOW}Сгенерированный файл:${RESET}"
        echo -e "  ${CYAN}$PROJECT_ROOT/CHANGELOG.md${RESET}"
        echo ""
        echo -e "${DIM}Первые 20 строк:${RESET}"
        head -20 "$PROJECT_ROOT/CHANGELOG.md"
    fi

    sleep "$PAUSE"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Итог
# ─────────────────────────────────────────────────────────────────────────────
divider
echo -e "${BOLD}${GREEN}Демо завершено!${RESET}"
echo ""
echo -e "${BOLD}Что было продемонстрировано:${RESET}"
echo ""
echo -e "  ${GREEN}✓${RESET} Ассистент сам выбирает нужные файлы без указания путей"
echo -e "  ${GREEN}✓${RESET} Поиск по нескольким файлам одновременно (grep_in_files)"
echo -e "  ${GREEN}✓${RESET} Проверка инвариантов кодовой базы"
echo -e "  ${GREEN}✓${RESET} Генерация документации с сохранением файла"
echo ""
echo -e "${BOLD}Команды для самостоятельного запуска:${RESET}"
echo ""
echo -e "  ${CYAN}# Интерактивный режим${RESET}"
echo -e "  ${BOLD}llm-cli --file-assistant${RESET}"
echo ""
echo -e "  ${CYAN}# Одна задача${RESET}"
echo -e "  ${BOLD}llm-cli --file-assistant --file-goal \"найди все TODO в коде\"${RESET}"
echo ""
echo -e "  ${CYAN}# Отдельные сценарии демо${RESET}"
echo -e "  ${BOLD}SCENARIO=1 bash scripts/demo_file_assistant.sh${RESET}"
echo -e "  ${BOLD}SCENARIO=2 bash scripts/demo_file_assistant.sh${RESET}"
echo -e "  ${BOLD}SCENARIO=3 bash scripts/demo_file_assistant.sh${RESET}"
echo ""
divider
