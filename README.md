# LLM CLI Chat

Терминальный клиент для [OpenRouter API](https://openrouter.ai/) с красивым выводом через Rich, бенчмарком моделей и моделью-судьёй.

## Требования

- Python 3.11+
- API ключ [OpenRouter](https://openrouter.ai/keys)

## Установка

```bash
# Клонировать репозиторий
git clone https://github.com/your-user/llm-api-demo.git
cd llm-api-demo

# Создать виртуальное окружение и установить зависимости
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -e .
```

## Запуск

```bash
# Интерактивный чат (при первом запуске попросит API-ключ)
python -m llm_cli

# Или через установленную команду
llm-cli

# Через переменную окружения
export OPENROUTER_API_KEY="sk-or-v1-ваш-ключ"   # Linux/macOS
set OPENROUTER_API_KEY=sk-or-v1-ваш-ключ         # Windows CMD
python -m llm_cli
```

### Бенчмарк моделей

```bash
# Сравнить модели с промптом по умолчанию
python -m llm_cli --compare

# Свой промпт и температура
python -m llm_cli --compare --prompt "Что такое рекурсия?" --temp 0.5
```

## Команды в чате

| Команда | Описание |
|---|---|
| `/temp 0.7` | Установить температуру генерации |
| `/compare` | Запустить бенчмарк-сравнение моделей |
| `/model deepseek/deepseek-chat-v3.1` | Сменить модель |
| `exit` / `quit` | Выйти |

Ввод сообщения: напишите текст и нажмите Enter дважды для отправки.

## Конфигурация

При первом запуске создаётся конфиг-файл:

- **Linux/macOS:** `~/.config/llm-cli/config.toml`
- **Windows:** `%APPDATA%\llm-cli\config.toml`

```toml
[general]
api_key = "sk-or-v1-..."
default_model = "deepseek/deepseek-chat-v3.1"
temperature = 1.0
benchmark_prompt = ""

[[models]]
id = "meta-llama/llama-3.3-70b-instruct"
tier = "Слабая (дешёвая)"
display_name = "Llama 3.3 70B"
input_price_per_million = 0.1
output_price_per_million = 0.32
url = "https://openrouter.ai/meta-llama/llama-3.3-70b-instruct"
```

## Структура проекта

```
src/llm_cli/
  __init__.py       — пакет
  __main__.py       — точка входа, argparse
  api.py            — HTTP-клиент OpenRouter (httpx)
  benchmark.py      — бенчмарк моделей + модель-судья
  chat.py           — интерактивный чат-цикл
  config.py         — загрузка/сохранение TOML-конфига
  display.py        — Rich-отрисовка (таблицы, панели, спиннеры)
  models.py         — Pydantic-модели данных
```
