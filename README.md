# LLM API Demo

Минимальное CLI-приложение на Kotlin, которое отправляет запрос в LLM через [OpenRouter API](https://openrouter.ai/) и выводит ответ в консоль.

## Требования

- Java 17+
- API ключ [OpenRouter](https://openrouter.ai/keys)

## Запуск

```bash
# Вариант 1: через переменную окружения
export OPENROUTER_API_KEY="sk-or-v1-ваш-ключ"
./gradlew run --console=plain

# Вариант 2: ключ будет запрошен при старте
./gradlew run --console=plain
```

### Выбор модели

По умолчанию используется `openai/gpt-3.5-turbo`. Можно указать другую:

```bash
export LLM_MODEL="anthropic/claude-3-haiku"
./gradlew run --console=plain
```

Список моделей: https://openrouter.ai/models

## Структура

```
src/main/kotlin/Main.kt   — весь код в одном файле (~100 строк)
build.gradle.kts           — Gradle конфигурация с зависимостями
```
