#!/usr/bin/env bash
# Демо-скрипт для записи видео: AI Daily Digest
# Запуск: bash scripts/demo_daily_digest.sh
#
# Переменные окружения:
#   OPENROUTER_API_KEY  — ключ OpenRouter
#   USE_LOCAL=1         — использовать локальный llama.cpp (ключ не нужен)
#   LOCAL_URL           — URL локального сервера (default: http://127.0.0.1:8081/v1/chat/completions)

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
DIM="\033[2m"
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
    # По умолчанию — локальный режим (USE_LOCAL=1)
    if [[ "${USE_LOCAL:-1}" == "1" ]]; then
        return 0
    fi
    if [[ -z "$OPENROUTER_API_KEY" ]]; then
        echo -e "${RED}ERROR: OPENROUTER_API_KEY не задан.${RESET}"
        echo -e "Варианты:"
        echo -e "  ${CYAN}export OPENROUTER_API_KEY=sk-or-...${RESET}   (OpenRouter)"
        echo -e "  ${CYAN}USE_LOCAL=1 bash scripts/demo_daily_digest.sh${RESET}  (локальный llama.cpp)"
        exit 1
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. Заголовок
# ─────────────────────────────────────────────────────────────────────────────
clear
header "AI Daily Digest — ДЕМО"
echo -e "${BOLD}Задача:${RESET}"
echo -e "  Каждое утро разработчик запускает одну команду"
echo -e "  и получает AI-саммари того, что произошло в репозитории за 24 часа."
echo ""
echo -e "${BOLD}Что будет показано:${RESET}"
echo -e "  ${CYAN}1.${RESET} Git-активность за последние 24 часа"
echo -e "  ${CYAN}2.${RESET} Статистика TODO/FIXME/HACK в коде"
echo -e "  ${CYAN}3.${RESET} Запрос к LLM — анализ и дайджест"
echo -e "  ${CYAN}4.${RESET} Структурированный отчёт: что сделано, риски, фокус на сегодня"
sleep 3

# ─────────────────────────────────────────────────────────────────────────────
# 2. Показываем git log (входные данные)
# ─────────────────────────────────────────────────────────────────────────────
header "1. Git-активность за последние 24 часа"
echo -e "${YELLOW}\$ git log --oneline --since=\"24 hours ago\" --no-merges${RESET}"
echo ""
cd "$PROJECT_ROOT"
COMMITS=$(git log --oneline --since="24 hours ago" --no-merges 2>/dev/null || true)
if [[ -n "$COMMITS" ]]; then
    echo "$COMMITS"
else
    echo -e "${DIM}(коммитов за 24ч нет — AI всё равно даст рекомендации)${RESET}"
fi
sleep 2

echo ""
echo -e "${YELLOW}\$ git diff --stat HEAD~1 HEAD${RESET}"
echo ""
git diff --stat HEAD~1 HEAD 2>/dev/null || echo -e "${DIM}(нет предыдущего коммита)${RESET}"
sleep 2

# ─────────────────────────────────────────────────────────────────────────────
# 3. Показываем TODO/FIXME статистику
# ─────────────────────────────────────────────────────────────────────────────
header "2. Технический долг: TODO / FIXME / HACK"
echo -e "${YELLOW}\$ rg --glob='*.py' --glob='!.venv/**' -c 'TODO|FIXME|HACK' src/ | head -10${RESET}"
echo ""
if command -v rg &>/dev/null; then
    rg --glob='*.py' --glob='!.venv/**' --glob='!venv/**' -c -i 'TODO|FIXME|HACK' \
        "$PROJECT_ROOT/src" 2>/dev/null | sort -t: -k2 -rn | head -10 \
        || echo -e "${DIM}(ничего не найдено)${RESET}"
else
    grep -rn --include='*.py' --exclude-dir='.venv' --exclude-dir='venv' \
        -i -c 'TODO\|FIXME\|HACK' "$PROJECT_ROOT/src" 2>/dev/null \
        | grep -v ':0$' | sort -t: -k2 -rn | head -10 \
        || echo -e "${DIM}(ничего не найдено)${RESET}"
fi
sleep 2

# ─────────────────────────────────────────────────────────────────────────────
# 4. Запускаем AI дайджест
# ─────────────────────────────────────────────────────────────────────────────
header "3. Запуск AI Daily Digest"
check_api_key

if [[ "${USE_LOCAL:-1}" == "1" ]]; then
    _LOCAL_URL="${LOCAL_URL:-http://127.0.0.1:8081/v1/chat/completions}"
    _MODE_LABEL="LOCAL llama.cpp ($_LOCAL_URL)"
else
    _MODE_LABEL="OpenRouter"
fi

echo -e "${YELLOW}Режим:${RESET} ${CYAN}$_MODE_LABEL${RESET}"
echo ""
echo -e "${YELLOW}Команда:${RESET}"
echo -e "  ${CYAN}python -m llm_cli --daily-digest${RESET}"
echo ""
echo -e "${BOLD}Что делает команда:${RESET}"
echo -e "  1. Собирает git log --since='24 hours ago'"
echo -e "  2. Ищет TODO/FIXME/HACK в *.py файлах"
echo -e "  3. Отправляет данные в LLM"
echo -e "  4. Выводит структурированный дайджест"
echo ""
echo -e "${YELLOW}Запускаем...${RESET}"
sleep 2

cd "$PROJECT_ROOT"

if [[ "${USE_LOCAL:-1}" == "1" ]]; then
    USE_LOCAL=1 LOCAL_URL="${LOCAL_URL:-http://127.0.0.1:8081/v1/chat/completions}" \
    "$PYTHON" -m llm_cli --daily-digest
else
    OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
    "$PYTHON" -m llm_cli --daily-digest
fi

# ─────────────────────────────────────────────────────────────────────────────
# 5. Итог
# ─────────────────────────────────────────────────────────────────────────────
divider
echo -e "${BOLD}${GREEN}Демо завершено!${RESET}"
echo ""
echo -e "${BOLD}Как использовать в работе:${RESET}"
echo -e "  ${CYAN}1.${RESET} Добавить в cron — запускать каждое утро:"
echo -e "     ${DIM}0 9 * * * cd /path/to/repo && python -m llm_cli --daily-digest${RESET}"
echo -e "  ${CYAN}2.${RESET} Запускать вручную перед standup:"
echo -e "     ${DIM}python -m llm_cli --daily-digest${RESET}"
echo -e "  ${CYAN}3.${RESET} Или через этот скрипт (с красивым вводом):"
echo -e "     ${DIM}bash scripts/demo_daily_digest.sh${RESET}"
echo ""
echo -e "Модуль: ${CYAN}src/llm_cli/daily_digest.py${RESET}"
divider
