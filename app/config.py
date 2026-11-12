"""Конфигурация приложения: пути, настройки pipeline, локальное хранилище.

Все настройки можно переопределить через переменные окружения (префикс ``AVTO_``)
или через файл ``data/settings.json``, который редактируется из веб-интерфейса.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields, asdict
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SETTINGS_FILE = DATA_DIR / "settings.json"

ENV_PREFIX = "AVTO_"


def _env(name: str, default: str) -> str:
    return os.environ.get(ENV_PREFIX + name, default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "on", "да"}


@dataclass
class Settings:
    """Настройки пайплайна и путей."""

    # --- рабочие папки -------------------------------------------------
    input_dir: str = "input"
    temp_dir: str = "temp"
    output_dir: str = "output"

    # --- локальный LLM (OpenAI-совместимый) -----------------------------
    llm_base_url: str = "http://127.0.0.1:1234/v1"
    llm_api_key: str = ""
    llm_model: str = "local-model"
    llm_timeout: float = 240.0
    llm_max_tokens: int = 4096

    # --- распознавание речи --------------------------------------------
    whisper_model: str = "medium"
    whisper_device: str = "auto"  # auto | cuda | cpu
    whisper_compute_type: str = "auto"  # auto | int8_float16 | int8 | float16
    whisper_language: str = "ru"
    transcribe_backend: str = "auto"  # auto | faster-whisper | stub

    # --- монтаж ---------------------------------------------------------
    silence_margin: float = 0.2
    loudnorm_i: float = -14.0
    loudnorm_tp: float = -1.5
    loudnorm_lra: float = 11.0
    keep_temp: bool = False

    # --- кодирование ----------------------------------------------------
    video_encoder: str = "auto"  # auto | h264_nvenc | libx264
    nvenc_preset: str = "p4"
    nvenc_cq: int = 20
    audio_bitrate: str = "192k"

    # --- шортсы ---------------------------------------------------------
    short_count: int = 3
    short_min_duration: float = 30.0
    short_max_duration: float = 55.0

    # --- сервис ---------------------------------------------------------
    max_upload_mb: int = 4096
    auto_watch: bool = False

    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls) -> "Settings":
        values: dict[str, Any] = {}
        for f in fields(cls):
            default = f.default
            name = f.name.upper()
            if isinstance(default, bool):
                values[f.name] = _env_bool(name, default)
            elif isinstance(default, int) and not isinstance(default, bool):
                values[f.name] = _env_int(name, default)
            elif isinstance(default, float):
                values[f.name] = _env_float(name, default)
            else:
                values[f.name] = _env(name, str(default))
        return cls(**values)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def update(self, payload: dict[str, Any]) -> None:
        """Обновить только известные поля (без валидации типов)."""
        known = {f.name for f in fields(self)}
        for key, value in (payload or {}).items():
            if key in known and value is not None:
                setattr(self, key, value)

    def save(self, path: Path | None = None) -> None:
        target = path or SETTINGS_FILE
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(target)

    @classmethod
    def load(cls) -> "Settings":
        settings = cls.from_env()
        if SETTINGS_FILE.exists():
            try:
                payload = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return settings
            if isinstance(payload, dict):
                settings.update(payload)
        return settings

    # ------------------------------------------------------------------
    def resolve_dir(self, name: str) -> Path:
        """Путь относительно корня проекта (абсолютные пути не трогаем)."""
        raw = Path(getattr(self, name)).expanduser()
        if not raw.is_absolute():
            raw = BASE_DIR / raw
        return raw.resolve()

    @property
    def input_path(self) -> Path:
        return self.resolve_dir("input_dir")

    @property
    def temp_path(self) -> Path:
        return self.resolve_dir("temp_dir")

    @property
    def output_path(self) -> Path:
        return self.resolve_dir("output_dir")

    def ensure_dirs(self) -> None:
        for path in (self.input_path, self.temp_path, self.output_path, DATA_DIR):
            path.mkdir(parents=True, exist_ok=True)
