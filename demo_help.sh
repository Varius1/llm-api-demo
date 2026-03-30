#!/usr/bin/env bash
# Демо-скрипт для записи видео: Developer Assistant (/help)
# Запуск: bash demo_help.sh

set -e

# Автоматически выбираем Python из venv если он есть
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/.venv/bin/python" ]]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
else
    PYTHON="python"
fi

# Путь к файлу истории чата
HISTORY_FILE="$HOME/.config/llm-cli/history.json"

BOLD="\033[1m"
CYAN="\033[1;36m"
GREEN="\033[1;32m"
YELLOW="\033[1;33m"
RESET="\033[0m"

divider() {
    echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n"
}

header() {
    divider
    echo -e "${BOLD}${GREEN}$1${RESET}"
    divider
}

# Очищаем историю чата чтобы демо было чистым
if [[ -f "$HISTORY_FILE" ]]; then
    mv "$HISTORY_FILE" "${HISTORY_FILE}.demo_backup"
fi
restore_history() {
    if [[ -f "${HISTORY_FILE}.demo_backup" ]]; then
        mv "${HISTORY_FILE}.demo_backup" "$HISTORY_FILE"
    fi
}
trap restore_history EXIT

# ─────────────────────────────────────────────────────────────────────────────
# 1. Показываем текущий git-контекст
# ─────────────────────────────────────────────────────────────────────────────
header "1. Git-контекст проекта"
echo -e "${YELLOW}Команда:${RESET} git branch && git log -1 --oneline"
echo ""
git branch
git log -1 --oneline
sleep 1

# ─────────────────────────────────────────────────────────────────────────────
# 2. /help без аргументов — справочник команд
# ─────────────────────────────────────────────────────────────────────────────
header "2. /help — справочник команд"
echo -e "${YELLOW}Ввод:${RESET} /help"
echo ""
# Двойной пустой Enter завершает multiline-ввод, затем exit выходит из чата
printf '/help\n\n\nexit\n' | "$PYTHON" -m llm_cli 2>/dev/null
sleep 1

# ─────────────────────────────────────────────────────────────────────────────
# 3. /help с конкретным вопросом о MCP
# ─────────────────────────────────────────────────────────────────────────────
header "3. /help — вопрос о MCP-инструментах"
echo -e "${YELLOW}Ввод:${RESET} /help Какие MCP-инструменты есть в проекте?"
echo ""
printf '/help Какие MCP-инструменты есть в проекте?\n\n\nexit\n' | "$PYTHON" -m llm_cli 2>/dev/null
sleep 1

# ─────────────────────────────────────────────────────────────────────────────
# 4. /help о RAG
# ─────────────────────────────────────────────────────────────────────────────
header "4. /help — вопрос о RAG-системе"
echo -e "${YELLOW}Ввод:${RESET} /help Как работает RAG в этом проекте?"
echo ""
printf '/help Как работает RAG в этом проекте?\n\n\nexit\n' | "$PYTHON" -m llm_cli 2>/dev/null
sleep 1

# ─────────────────────────────────────────────────────────────────────────────
# 5. /help о текущей git-ветке
# ─────────────────────────────────────────────────────────────────────────────
header "5. /help — git-контекст через ассистента"
echo -e "${YELLOW}Ввод:${RESET} /help На какой я сейчас ветке и что последнее было сделано?"
echo ""
printf '/help На какой я сейчас ветке и что последнее было сделано?\n\n\nexit\n' | "$PYTHON" -m llm_cli 2>/dev/null

divider
echo -e "${BOLD}${GREEN}Демо завершено!${RESET}"
echo -e "Команда ${CYAN}/help${RESET} доступна в обычном чате: ${CYAN}$PYTHON -m llm_cli${RESET}"
divider
