from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


APP_DIR = Path.home() / ".pub_reader"
CONFIG_PATH = APP_DIR / "config.json"
LIBRARY_DIR = APP_DIR / "library"


@dataclass
class AppConfig:
    api_base_url: str = "https://api.deepseek.com/chat/completions"
    model_name: str = "deepseek-v4-pro"
    library_dir: str = str(LIBRARY_DIR)


def load_config() -> AppConfig:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        config = AppConfig()
        save_config(config)
        return config

    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return AppConfig(**{**asdict(AppConfig()), **data})


def save_config(config: AppConfig) -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
