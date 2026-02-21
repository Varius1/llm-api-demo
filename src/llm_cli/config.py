"""Загрузка и сохранение конфигурации в TOML-файле."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

from platformdirs import user_config_dir
from rich.console import Console
from rich.prompt import Prompt

from .models import BENCHMARK_MODELS, DEFAULT_MODEL, ModelConfig

APP_NAME = "llm-cli"
CONFIG_FILENAME = "config.toml"

console = Console()


def _config_path() -> Path:
    return Path(user_config_dir(APP_NAME, appauthor=False, ensure_exists=True)) / CONFIG_FILENAME


def _serialize_toml(cfg: AppConfig) -> str:
    """Минимальная сериализация в TOML (без внешних зависимостей для записи)."""
    lines: list[str] = ["[general]"]
    lines.append(f'api_key = "{cfg.api_key}"')
    lines.append(f'default_model = "{cfg.default_model}"')
    lines.append(f"temperature = {cfg.temperature}")
    lines.append(f'benchmark_prompt = "{cfg.benchmark_prompt}"')
    lines.append("")

    for m in cfg.models:
        lines.append("[[models]]")
        lines.append(f'id = "{m.id}"')
        lines.append(f'tier = "{m.tier}"')
        lines.append(f'display_name = "{m.display_name}"')
        lines.append(f"input_price_per_million = {m.input_price_per_million}")
        lines.append(f"output_price_per_million = {m.output_price_per_million}")
        lines.append(f'url = "{m.url}"')
        lines.append("")

    return "\n".join(lines) + "\n"


class AppConfig:
    """Конфигурация приложения, загружаемая из TOML."""

    def __init__(
        self,
        api_key: str = "",
        default_model: str = DEFAULT_MODEL,
        temperature: float = 1.0,
        benchmark_prompt: str = "",
        models: list[ModelConfig] | None = None,
    ):
        self.api_key = api_key
        self.default_model = default_model
        self.temperature = temperature
        self.benchmark_prompt = benchmark_prompt
        self.models = models if models is not None else list(BENCHMARK_MODELS)

    def save(self) -> None:
        path = _config_path()
        path.write_text(_serialize_toml(self), encoding="utf-8")

    @classmethod
    def load(cls) -> AppConfig:
        path = _config_path()
        if not path.exists():
            return cls()

        with open(path, "rb") as f:
            data = tomllib.load(f)

        general = data.get("general", {})
        models_raw = data.get("models", [])

        models = [
            ModelConfig(
                id=m["id"],
                tier=m.get("tier", ""),
                display_name=m.get("display_name", m["id"]),
                input_price_per_million=m.get("input_price_per_million", 0.0),
                output_price_per_million=m.get("output_price_per_million", 0.0),
                url=m.get("url", ""),
            )
            for m in models_raw
        ]

        return cls(
            api_key=general.get("api_key", ""),
            default_model=general.get("default_model", DEFAULT_MODEL),
            temperature=general.get("temperature", 1.0),
            benchmark_prompt=general.get("benchmark_prompt", ""),
            models=models if models else list(BENCHMARK_MODELS),
        )


def ensure_config() -> AppConfig:
    """Загрузить конфиг. Если API-ключа нет — запустить интерактивную настройку."""
    import os

    cfg = AppConfig.load()

    env_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if env_key:
        cfg.api_key = env_key

    if cfg.api_key:
        return cfg

    console.print()
    console.print("[bold]Первый запуск — настройка LLM CLI[/bold]", style="cyan")
    console.print()
    console.print(
        "Для работы нужен API-ключ OpenRouter.\n"
        "Получить: [link=https://openrouter.ai/keys]https://openrouter.ai/keys[/link]"
    )
    console.print()

    api_key = Prompt.ask("[bold yellow]Введите ваш OpenRouter API ключ[/bold yellow]").strip()
    if not api_key:
        console.print("[red]API ключ не введён. Выход.[/red]")
        sys.exit(1)

    cfg.api_key = api_key

    default_model = Prompt.ask(
        "Модель по умолчанию",
        default=cfg.default_model,
    )
    cfg.default_model = default_model

    cfg.save()
    config_path = _config_path()
    console.print(f"\n[green]Конфиг сохранён:[/green] {config_path}")
    console.print()

    return cfg
