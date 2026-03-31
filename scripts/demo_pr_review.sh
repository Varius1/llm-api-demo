#!/usr/bin/env bash
# Демо-скрипт для записи видео: AI Code Review Pipeline
# Запуск: bash scripts/demo_pr_review.sh
#
# Режим SIMULATE=1 — не обращается к GitHub API,
# читает локальный файл с намеренно «сломанным» кодом.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -f "$PROJECT_ROOT/.venv/bin/python" ]]; then
    PYTHON="$PROJECT_ROOT/.venv/bin/python"
else
    PYTHON="python"
fi

BOLD="\033[1m"
CYAN="\033[1;36m"
GREEN="\033[1;32m"
YELLOW="\033[1;33m"
RED="\033[1;31m"
RESET="\033[0m"

divider() {
    echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n"
}

header() {
    divider
    echo -e "${BOLD}${GREEN}$1${RESET}"
    divider
}

check_api_key() {
    # В локальном режиме ключ не нужен
    if [[ "${USE_LOCAL:-0}" == "1" ]]; then
        return 0
    fi
    if [[ -z "$OPENROUTER_API_KEY" ]]; then
        echo -e "${RED}ERROR: OPENROUTER_API_KEY не задан.${RESET}"
        echo -e "Варианты:"
        echo -e "  ${CYAN}export OPENROUTER_API_KEY=sk-or-...${RESET}   (OpenRouter)"
        echo -e "  ${CYAN}USE_LOCAL=1 bash scripts/demo_pr_review.sh${RESET}  (локальный llama.cpp)"
        exit 1
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. Заголовок демо
# ─────────────────────────────────────────────────────────────────────────────
clear
header "🤖  AI Code Review Pipeline — ДЕМО"
echo -e "${BOLD}Что будет показано:${RESET}"
echo -e "  ${CYAN}1.${RESET} Целевой файл с кодом (потенциальные баги)"
echo -e "  ${CYAN}2.${RESET} Формирование git diff"
echo -e "  ${CYAN}3.${RESET} Запуск AI ревью (LLM + RAG)"
echo -e "  ${CYAN}4.${RESET} Структурированный вывод ревью"
sleep 2

# ─────────────────────────────────────────────────────────────────────────────
# 2. Показываем файл для ревью
# ─────────────────────────────────────────────────────────────────────────────
header "1. Файл с кодом для ревью: buggy_calc.py"
echo -e "${YELLOW}$ cat buggy_calc.py${RESET}"
echo ""
cat "$PROJECT_ROOT/buggy_calc.py"
sleep 2

# ─────────────────────────────────────────────────────────────────────────────
# 3. Показываем как выглядит diff
# ─────────────────────────────────────────────────────────────────────────────
header "2. Git diff (симуляция PR)"
echo -e "${YELLOW}# В реальном PR diff берётся из GitHub API${RESET}"
echo -e "${YELLOW}# Здесь симулируем: добавляем файл как новый${RESET}"
echo ""
echo -e "${CYAN}diff --git a/buggy_calc.py b/buggy_calc.py${RESET}"
echo -e "${CYAN}--- /dev/null${RESET}"
echo -e "${CYAN}+++ b/buggy_calc.py${RESET}"
echo -e "${CYAN}@@ -0,0 +1,11 @@${RESET}"
while IFS= read -r line; do
    echo -e "${GREEN}+${line}${RESET}"
done < "$PROJECT_ROOT/buggy_calc.py"
sleep 2

# ─────────────────────────────────────────────────────────────────────────────
# 4. Проверяем ключ и запускаем ревью
# ─────────────────────────────────────────────────────────────────────────────
header "3. Запуск AI Code Review"
check_api_key

# Определяем режим: локальный llama.cpp или OpenRouter
if [[ "${USE_LOCAL:-0}" == "1" ]]; then
    _LOCAL_URL="${LOCAL_URL:-http://127.0.0.1:8081/v1/chat/completions}"
    _MODE_LABEL="LOCAL llama.cpp ($_LOCAL_URL)"
    _MODE_CMD="USE_LOCAL=1 LOCAL_URL=$_LOCAL_URL"
else
    _MODE_LABEL="OpenRouter"
    _MODE_CMD="OPENROUTER_API_KEY=\$OPENROUTER_API_KEY"
fi

echo -e "${YELLOW}Режим:${RESET} ${CYAN}$_MODE_LABEL${RESET}"
echo ""
echo -e "${YELLOW}Команда:${RESET}"
echo -e "  ${CYAN}SIMULATE=1 SIMULATE_FILE=buggy_calc.py $_MODE_CMD python scripts/pr_review.py${RESET}"
echo ""
echo -e "${BOLD}Параметры:${RESET}"
echo -e "  SIMULATE=1          — локальный режим (без GitHub API)"
echo -e "  SIMULATE_FILE=...   — файл для анализа"
echo -e "  USE_RAG=1           — поиск по документации проекта (FAISS)"
if [[ "${USE_LOCAL:-0}" == "1" ]]; then
echo -e "  USE_LOCAL=1         — использовать локальный llama.cpp (без ключа API)"
fi
echo ""
echo -e "${YELLOW}Запускаем...${RESET}"
sleep 1

cd "$PROJECT_ROOT"

SIMULATE=1 \
SIMULATE_FILE="$PROJECT_ROOT/buggy_calc.py" \
USE_RAG=1 \
USE_LOCAL="${USE_LOCAL:-0}" \
LOCAL_URL="${LOCAL_URL:-http://127.0.0.1:8081/v1/chat/completions}" \
"$PYTHON" scripts/pr_review.py

# ─────────────────────────────────────────────────────────────────────────────
# 5. Итог
# ─────────────────────────────────────────────────────────────────────────────
divider
echo -e "${BOLD}${GREEN}Демо завершено!${RESET}"
echo ""
echo -e "${BOLD}В реальном PR (GitHub Actions):${RESET}"
echo -e "  ${CYAN}1.${RESET} Workflow запускается автоматически при открытии/обновлении PR"
echo -e "  ${CYAN}2.${RESET} Скрипт получает diff через GitHub API"
echo -e "  ${CYAN}3.${RESET} LLM генерирует ревью с RAG-контекстом"
echo -e "  ${CYAN}4.${RESET} Комментарий публикуется прямо в PR"
echo ""
echo -e "Workflow: ${CYAN}.github/workflows/pr_review.yml${RESET}"
echo -e "Скрипт:   ${CYAN}scripts/pr_review.py${RESET}"
divider
