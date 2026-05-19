from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


APP_DIR = Path.home() / ".pub_reader"
CONFIG_PATH = APP_DIR / "config.json"
OLD_LIBRARY_DIR = APP_DIR / "library"


def project_root() -> Path:
    """Return the cloned project root when the app runs from source or dist."""
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        for candidate in [exe_dir, *exe_dir.parents]:
            if (candidate / "pyproject.toml").exists():
                return candidate
        return exe_dir
    return Path(__file__).resolve().parents[1]


OUTPUT_DIR = project_root() / "output"
DEFAULT_FOLDER_NAME = "默认文件夹"
DEFAULT_FOLDER_DIR = OUTPUT_DIR / DEFAULT_FOLDER_NAME


@dataclass
class AppConfig:
    api_base_url: str = "https://api.deepseek.com/chat/completions"
    model_name: str = "deepseek-v4-pro"
    library_dir: str = str(OUTPUT_DIR)


def load_config() -> AppConfig:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_FOLDER_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        config = AppConfig()
        save_config(config)
        return config

    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config = AppConfig(**{**asdict(AppConfig()), **data})

    # Older builds stored papers under ~/.pub_reader/library. Move the default
    # config forward while preserving a user-edited custom library path.
    if Path(config.library_dir) == OLD_LIBRARY_DIR:
        config.library_dir = str(OUTPUT_DIR)
        save_config(config)
    Path(config.library_dir).mkdir(parents=True, exist_ok=True)
    (Path(config.library_dir) / DEFAULT_FOLDER_NAME).mkdir(parents=True, exist_ok=True)
    return config


def save_config(config: AppConfig) -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
