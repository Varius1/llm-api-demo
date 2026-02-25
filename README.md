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
| `/model google/gemma-2-9b-it` | Сменить модель |
| `/clear` | Очистить историю диалога (в памяти и в файле истории) |
| `/overflow 9000` | Отправить большой тестовый запрос в strict-режиме (`transforms: []`) |
| `exit` / `quit` | Выйти |

Ввод сообщения: напишите текст и нажмите Enter дважды для отправки.

## Конфигурация

При первом запуске создаётся конфиг-файл:

- **Linux/macOS:** `~/.config/llm-cli/config.toml`
- **Windows:** `%APPDATA%\llm-cli\config.toml`

История диалога сохраняется отдельно:

- **Linux/macOS:** `~/.config/llm-cli/history.json`
- **Windows:** `%APPDATA%\llm-cli\history.json`

При новом запуске история автоматически загружается, и диалог продолжается с прошлого контекста.

```toml
[general]
api_key = "sk-or-v1-..."
default_model = "google/gemma-2-9b-it"
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

## Токены и стоимость в чате

После каждого ответа CLI показывает:

- токены текущего запроса (оценка),
- токены всей истории (`prompt`, по данным API),
- токены ответа модели (`response`, по данным API),
- итог токенов (`total`) и стоимость за ход/сессию.

Это позволяет увидеть, как при длинном диалоге растут `prompt_tokens` и стоимость.

## Демонстрация переполнения контекста

Для задания можно воспроизвести реальное переполнение лимита модели:

```bash
python -m llm_cli
```

В чате:

1. `/model google/gemma-2-9b-it` (контекст ~8192 токена)
2. `/clear`
3. `/overflow 9000` (команда отправляется в strict-режиме с `transforms: []`)

Ожидаемое поведение: API вернёт `HTTP 400` с ошибкой переполнения контекста.

## Архитектура

```
CLI (chat.py)  →  Agent (agent.py)  →  HTTP-клиент (api.py)  →  OpenRouter API
   ввод/вывод      логика + история      транспорт               LLM
```

Агент (`Agent`) — отдельная сущность, инкапсулирующая логику диалога с LLM:
- хранит историю сообщений (контекст диалога)
- управляет параметрами генерации (модель, температура)
- поддерживает системный промпт
- не зависит от интерфейса — можно подключить к CLI, веб-серверу или боту

## Структура проекта

```
src/llm_cli/
  __init__.py       — пакет
  __main__.py       — точка входа, argparse
  agent.py          — LLM-агент (история диалога, инкапсуляция логики)
  api.py            — HTTP-клиент OpenRouter (httpx)
  benchmark.py      — бенчмарк моделей + модель-судья
  chat.py           — интерактивный чат-цикл (тонкая обёртка над агентом)
  config.py         — загрузка/сохранение TOML-конфига
  display.py        — Rich-отрисовка (таблицы, панели, спиннеры)
  models.py         — Pydantic-модели данных
```
