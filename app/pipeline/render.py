"""Рендер: нарезка, склейка, нормализация звука, адаптация под 16:9 и 9:16."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Sequence

from ..config import Settings
from ..ffmpeg_tools import (
    MediaError,
    aspect_filter,
    encoder_args,
    loudnorm_filter,
    probe,
    require_ffmpeg,
    resolve_encoder,
    run_cmd,
    trim_filter_graph,
)
from .edit import Interval

LOGGER = logging.getLogger("avto.render")

LogFn = Callable[[str], None]


class RenderError(RuntimeError):
    """Ошибка рендера."""


def _segments_from_intervals(intervals: Sequence[Interval]) -> list[tuple[float, float]]:
    return [(round(i.start, 3), round(i.end, 3)) for i in intervals if i.duration > 0.02]


def cut_and_concat(
    source: Path,
    intervals: Sequence[Interval],
    destination: Path,
    *,
    video_post: str = "",
    audio_filter: str = "",
    settings: Settings,
    encoder: str | None = None,
    log: LogFn | None = None,
    label: str = "",
    should_stop: Callable[[], bool] | None = None,
) -> Path:
    """Вырезает отрезки из ``source``, склеивает их и кодирует результат."""
    emit = log or (lambda message: LOGGER.info(message))
    segments = _segments_from_intervals(intervals)
    if not segments:
        raise RenderError("Нет ни одного сегмента для рендера")

    info = probe(source)
    has_audio = info.has_audio
    chosen_encoder = encoder or resolve_encoder(settings.video_encoder)
    fps = info.fps if info.fps and info.fps <= 120 else None

    filter_complex, mapping = trim_filter_graph(
        segments,
        audio_filter=audio_filter if has_audio else "",
        video_post=video_post,
    )

    prefix = f"{label} " if label else ""
    emit(
        f"{prefix}рендер: {len(segments)} фрагм., кодировщик {chosen_encoder}, "
        f"разрешение {info.width}x{info.height}"
    )

    def build(codec: str) -> list[str]:
        cmd = [require_ffmpeg(), "-y", "-hide_banner", "-nostdin", "-i", str(source)]
        cmd += ["-filter_complex", filter_complex]
        cmd += ["-map", mapping.split()[0]]
        if has_audio:
            cmd += ["-map", mapping.split()[1]]
        cmd += encoder_args(
            codec,
            nvenc_preset=settings.nvenc_preset,
            nvenc_cq=settings.nvenc_cq,
            audio_bitrate=settings.audio_bitrate,
            fps=fps,
        )
        cmd += [str(destination)]
        return cmd

    try:
        run_cmd(build(chosen_encoder), timeout=None, should_stop=should_stop)
    except MediaError as exc:
        if "отменена" in str(exc).lower():
            raise RenderError(str(exc)) from exc
        if chosen_encoder == "h264_nvenc":
            emit("h264_nvenc не сработал — повторяю на libx264 (программный кодировщик).")
            try:
                run_cmd(build("libx264"), timeout=None, should_stop=should_stop)
            except MediaError as fallback_exc:
                raise RenderError(str(fallback_exc)) from fallback_exc
        else:
            raise RenderError(str(exc)) from exc

    if not destination.exists() or destination.stat().st_size == 0:
        raise RenderError(f"Файл не создан: {destination}")
    emit(
        f"{prefix}готово: {destination.name} "
        f"({destination.stat().st_size / 1_048_576:.1f} МБ)"
    )
    return destination


def render_final(
    source: Path,
    keep: Sequence[Interval],
    destination: Path,
    aspect: str,
    settings: Settings,
    *,
    log: LogFn | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> Path:
    """Собирает чистовой ролик в выбранном формате."""
    emit = log or (lambda message: LOGGER.info(message))
    info = probe(source)
    video_filter = aspect_filter(aspect, info.width, info.height)
    audio = loudnorm_filter(
        settings.loudnorm_i, settings.loudnorm_tp, settings.loudnorm_lra
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    return cut_and_concat(
        source,
        keep,
        destination,
        video_post=video_filter,
        audio_filter=audio,
        settings=settings,
        log=emit,
        label=f"[{aspect}] ",
        should_stop=should_stop,
    )


def render_short(
    source: Path,
    short: Interval,
    destination: Path,
    aspect: str,
    settings: Settings,
    *,
    log: LogFn | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> Path:
    """Собирает вертикальный (или горизонтальный) короткий ролик."""
    emit = log or (lambda message: LOGGER.info(message))
    info = probe(source)
    video_filter = aspect_filter(aspect, info.width, info.height)
    audio = loudnorm_filter(
        settings.loudnorm_i, settings.loudnorm_tp, settings.loudnorm_lra
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    return cut_and_concat(
        source,
        [short],
        destination,
        video_post=video_filter,
        audio_filter=audio,
        settings=settings,
        log=emit,
        label=f"[short {aspect}] ",
        should_stop=should_stop,
    )
