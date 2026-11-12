"""Удаление тишины и пауз: auto-editor (основной) + ffmpeg (запасной)."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

from ..config import Settings
from ..ffmpeg_tools import CREATE_NO_WINDOW, MediaError, probe, require_ffmpeg, run_cmd
from .edit import Interval
from .render import cut_and_concat

LOGGER = logging.getLogger("avto.silence")

LogFn = Callable[[str], None]


def find_auto_editor() -> list[str] | None:
    """Команда запуска auto-editor (консольный скрипт или python -m)."""
    binary = shutil.which("auto-editor")
    if binary:
        return [binary]
    try:
        import auto_editor  # type: ignore  # noqa: F401

        return [sys.executable, "-m", "auto_editor"]
    except Exception:
        return None


def remove_silence(
    source: Path,
    destination: Path,
    *,
    margin: float = 0.2,
    settings: Settings,
    log: LogFn | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> Path:
    """Убирает паузы из видео. Возвращает путь к очищенному файлу."""
    emit = log or (lambda message: LOGGER.info(message))
    destination.parent.mkdir(parents=True, exist_ok=True)

    command = find_auto_editor()
    if command:
        cmd = command + [
            "--margin",
            f"{margin}sec",
            "-o",
            str(destination),
            str(source),
        ]
        emit("Запускаю auto-editor (удаление пауз)...")
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=CREATE_NO_WINDOW,
                timeout=3600,
            )
            if proc.returncode == 0 and destination.exists() and destination.stat().st_size > 0:
                emit("auto-editor: паузы удалены.")
                return destination
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
            emit("auto-editor не справился, переключаюсь на ffmpeg:\n" + "\n".join(tail))
        except (OSError, subprocess.TimeoutExpired) as exc:
            emit(f"auto-editor недоступен ({exc}), переключаюсь на ffmpeg.")

    return _remove_silence_ffmpeg(
        source,
        destination,
        margin=margin,
        settings=settings,
        log=emit,
        should_stop=should_stop,
    )


def _remove_silence_ffmpeg(
    source: Path,
    destination: Path,
    *,
    margin: float,
    settings: Settings,
    log: LogFn,
    should_stop: Callable[[], bool] | None = None,
) -> Path:
    """Запасной вариант: детект тишины средствами ffmpeg + склейка."""
    emit = log
    emit("Удаляю паузы средствами ffmpeg (silencedetect)...")
    intervals = detect_speech_intervals(source, margin=margin)
    if not intervals:
        emit("Речь не обнаружена — копирую исходник без изменений.")
        shutil.copy2(source, destination)
        return destination

    total = sum(i.duration for i in intervals)
    info = probe(source)
    emit(
        f"Найдено речевых участков: {len(intervals)} "
        f"({total:.1f} с из {info.duration:.1f} с)"
    )
    cut_and_concat(
        source,
        intervals,
        destination,
        settings=settings,
        log=emit,
        label="[silence] ",
        should_stop=should_stop,
    )
    return destination


def detect_speech_intervals(source: Path, *, margin: float = 0.2) -> list[Interval]:
    """Возвращает интервалы с речью (тишина вырезается с отступом margin)."""
    try:
        proc = run_cmd(
            [
                require_ffmpeg(),
                "-hide_banner",
                "-nostdin",
                "-i",
                str(source),
                "-af",
                "silencedetect=noise=-30dB:d=0.35",
                "-f",
                "null",
                "-",
            ],
            check=False,
        )
    except MediaError:
        return []

    text = (proc.stderr or "") + (proc.stdout or "")
    info = probe(source)
    duration = info.duration

    silences: list[tuple[float, float]] = []
    for match in re.finditer(
        r"silence_start:\s*(-?[\d.]+)\s*\|.*?silence_end:\s*([\d.]+)", text
    ):
        silences.append((float(match.group(1)), float(match.group(2))))

    if not silences:
        return [Interval(0.0, duration)] if duration else []

    speech: list[Interval] = []
    cursor = 0.0
    for start, end in silences:
        if start - cursor > 0.05:
            speech.append(Interval(cursor, max(cursor, start - margin)))
        cursor = max(cursor, end)
    if duration > cursor + 0.05:
        speech.append(Interval(cursor, duration))

    return [item for item in speech if item.duration > 0.05]
