"""Работа с ffmpeg/ffprobe: поиск бинарников, probe, выбор кодировщика.

Особенность окружения: у пользователя Windows + NVIDIA RTX 2060 SUPER, поэтому
основной режим — ``h264_nvenc``. Если видеокарты нет (или драйвер не установлен),
автоматически включается ``libx264``, чтобы приложение всё равно работало.
"""

from __future__ import annotations

import functools
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0


class MediaError(RuntimeError):
    """Ошибка при работе с ffmpeg."""


@dataclass
class MediaInfo:
    path: Path
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    has_audio: bool = False
    has_video: bool = False
    raw: dict[str, Any] | None = None

    @property
    def aspect(self) -> float:
        if self.width and self.height:
            return self.width / self.height
        return 16 / 9

    @property
    def is_vertical(self) -> bool:
        return self.aspect < 1.0


# ---------------------------------------------------------------------------
# Поиск бинарников
# ---------------------------------------------------------------------------
def _bundled_ffmpeg() -> str | None:
    """ffmpeg, поставляемый пакетом imageio-ffmpeg (запасной вариант)."""
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # pragma: no cover - опциональная зависимость
        return None


@functools.lru_cache(maxsize=1)
def find_ffmpeg() -> str | None:
    env = os.environ.get("AVTO_FFMPEG") or os.environ.get("FFMPEG_BINARY")
    if env and Path(env).exists():
        return env
    found = shutil.which("ffmpeg")
    if found:
        return found
    return _bundled_ffmpeg()


@functools.lru_cache(maxsize=1)
def find_ffprobe() -> str | None:
    env = os.environ.get("AVTO_FFPROBE") or os.environ.get("FFPROBE_BINARY")
    if env and Path(env).exists():
        return env
    found = shutil.which("ffprobe")
    if found:
        return found
    # imageio-ffmpeg не содержит ffprobe, но часто лежит рядом с ffmpeg
    ffmpeg = find_ffmpeg()
    if ffmpeg:
        sibling = Path(ffmpeg).with_name(
            "ffprobe.exe" if sys.platform == "win32" else "ffprobe"
        )
        if sibling.exists():
            return str(sibling)
    return None


def require_ffmpeg() -> str:
    binary = find_ffmpeg()
    if not binary:
        raise MediaError(
            "ffmpeg не найден. Установите ffmpeg и добавьте его в PATH "
            "(https://www.gyan.dev/ffmpeg/builds/) либо выполните: pip install imageio-ffmpeg"
        )
    return binary


# ---------------------------------------------------------------------------
# Запуск процессов
# ---------------------------------------------------------------------------
def run_cmd(
    cmd: Sequence[str],
    *,
    timeout: float | None = None,
    check: bool = True,
    cwd: Path | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> subprocess.CompletedProcess:
    """Запускает команду, возвращая CompletedProcess с текстовым выводом.

    ``should_stop`` — необязательный колбэк: если он вернёт True, процесс
    останавливается (используется для отмены задачи из интерфейса).
    """
    try:
        process = subprocess.Popen(
            [str(c) for c in cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(cwd) if cwd else None,
            creationflags=CREATE_NO_WINDOW,
        )
    except FileNotFoundError as exc:
        raise MediaError(f"Не найден исполняемый файл: {cmd[0]}") from exc

    started = time.time()
    while True:
        try:
            stdout, stderr = process.communicate(timeout=0.5)
            break
        except subprocess.TimeoutExpired:
            if should_stop is not None and should_stop():
                _terminate(process)
                raise MediaError("Операция отменена пользователем")
            if timeout is not None and time.time() - started > timeout:
                _terminate(process)
                raise MediaError(
                    f"Команда превысила лимит времени: {' '.join(str(c) for c in cmd[:3])}..."
                )

    result = subprocess.CompletedProcess(
        args=[str(c) for c in cmd],
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
    )
    if check and result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-15:]
        raise MediaError(
            f"ffmpeg завершился с кодом {result.returncode}:\n" + "\n".join(tail)
        )
    return result


def _terminate(process: subprocess.Popen) -> None:
    try:
        process.terminate()
        process.wait(timeout=5)
    except Exception:  # pragma: no cover
        try:
            process.kill()
        except Exception:
            pass


def ffmpeg_version() -> str:
    try:
        proc = run_cmd([require_ffmpeg(), "-hide_banner", "-version"], check=False)
        first = (proc.stdout or "").splitlines()
        return first[0] if first else "unknown"
    except MediaError:
        return "unknown"


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------
_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)")
_VIDEO_RE = re.compile(
    r"Stream #\d+:\d+.*?:\s*Video:.*?(\d{2,5})x(\d{2,5})", re.IGNORECASE
)
_FPS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*fps", re.IGNORECASE)
_AUDIO_RE = re.compile(r"Stream #\d+:\d+.*?:\s*Audio:", re.IGNORECASE)


def _parse_fraction(value: str) -> float:
    try:
        if "/" in value:
            num, den = value.split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def probe(path: Path | str) -> MediaInfo:
    """Возвращает метаданные видео (ffprobe, при отсутствии — разбор вывода ffmpeg)."""
    path = Path(path)
    if not path.exists():
        raise MediaError(f"Файл не найден: {path}")

    ffprobe = find_ffprobe()
    if ffprobe:
        proc = run_cmd(
            [
                ffprobe,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            check=False,
        )
        try:
            data = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError:
            data = {}
        if data.get("streams"):
            info = MediaInfo(path=path, raw=data)
            fmt = data.get("format", {})
            info.duration = float(fmt.get("duration") or 0.0)
            for stream in data.get("streams", []):
                kind = stream.get("codec_type")
                if kind == "video" and not info.has_video:
                    info.has_video = True
                    info.width = int(stream.get("width") or 0)
                    info.height = int(stream.get("height") or 0)
                    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0"
                    info.fps = _parse_fraction(str(rate))
                elif kind == "audio":
                    info.has_audio = True
            if not info.duration:
                for stream in data.get("streams", []):
                    if stream.get("duration"):
                        info.duration = float(stream["duration"])
                        break
            return info

    # Запасной путь: разбираем stderr от `ffmpeg -i`
    proc = run_cmd([require_ffmpeg(), "-hide_banner", "-i", str(path)], check=False)
    text = (proc.stderr or "") + (proc.stdout or "")
    info = MediaInfo(path=path, raw={"parsed": text})
    match = _DURATION_RE.search(text)
    if match:
        h, m, s = match.groups()
        info.duration = int(h) * 3600 + int(m) * 60 + float(s)
    video = _VIDEO_RE.search(text)
    if video:
        info.has_video = True
        info.width = int(video.group(1))
        info.height = int(video.group(2))
    fps = _FPS_RE.search(text)
    if fps:
        info.fps = float(fps.group(1))
    info.has_audio = bool(_AUDIO_RE.search(text))
    return info


# ---------------------------------------------------------------------------
# Аппаратные возможности
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def nvenc_supported() -> bool:
    """Есть ли в ffmpeg кодировщик h264_nvenc (само наличие GPU не гарантирует)."""
    try:
        proc = run_cmd(
            [require_ffmpeg(), "-hide_banner", "-encoders"], check=False
        )
    except MediaError:
        return False
    return "h264_nvenc" in (proc.stdout or "")


@functools.lru_cache(maxsize=1)
def cuda_available() -> bool:
    """Доступен ли CUDA (для faster-whisper)."""
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return True
    except Exception:
        pass
    return shutil.which("nvidia-smi") is not None


@functools.lru_cache(maxsize=1)
def gpu_present() -> bool:
    """Есть ли в системе NVIDIA GPU (драйвер nvidia-smi или torch с CUDA)."""
    if shutil.which("nvidia-smi"):
        return True
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def nvenc_usable() -> bool:
    """Можно ли реально кодировать через видеокарту (драйвер + кодировщик)."""
    return nvenc_supported() and gpu_present()


@functools.lru_cache(maxsize=1)
def gpu_name() -> str:
    smi = shutil.which("nvidia-smi")
    if smi:
        proc = run_cmd(
            [smi, "--query-gpu=name", "--format=csv,noheader"], check=False
        )
        lines = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
        if lines:
            return lines[0]
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "CPU"


def resolve_encoder(preference: str = "auto") -> str:
    """Выбирает кодировщик: auto → h264_nvenc при наличии, иначе libx264."""
    if preference == "h264_nvenc":
        return "h264_nvenc"
    if preference == "libx264":
        return "libx264"
    return "h264_nvenc" if nvenc_usable() else "libx264"


def encoder_args(
    encoder: str,
    *,
    nvenc_preset: str = "p4",
    nvenc_cq: int = 20,
    audio_bitrate: str = "192k",
    fps: float | None = None,
) -> list[str]:
    """Аргументы кодирования видео и аудио для ffmpeg."""
    args: list[str] = ["-c:v", encoder]
    if encoder == "h264_nvenc":
        args += ["-preset", nvenc_preset, "-rc", "vbr", "-cq", str(nvenc_cq),
                 "-b:v", "0", "-pix_fmt", "yuv420p"]
    else:
        crf = max(18, min(28, int(round(21 + (nvenc_cq - 20) * 0.5))))
        args += ["-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p"]
    if fps and fps > 0:
        args += ["-r", f"{fps:.3f}"]
    args += ["-c:a", "aac", "-b:a", audio_bitrate, "-ar", "48000"]
    args += ["-movflags", "+faststart"]
    return args


# ---------------------------------------------------------------------------
# Фильтры
# ---------------------------------------------------------------------------
def loudnorm_filter(loudnorm_i: float = -14.0, loudnorm_tp: float = -1.5,
                     loudnorm_lra: float = 11.0) -> str:
    return f"loudnorm=I={loudnorm_i}:TP={loudnorm_tp}:LRA={loudnorm_lra}"


def aspect_filter(
    target: str,
    src_width: int = 1920,
    src_height: int = 1080,
) -> str:
    """Собирает цепочку фильтров под целевой формат.

    ``9:16`` — кроп по центру (как в ТЗ) с масштабированием до 1080x1920.
    ``16:9`` — если исходник вертикальный, мягкий blur-фон + центр,
    иначе — кроп по центру с масштабированием до 1920x1080.
    """
    if target == "9:16":
        return (
            "scale=1080:1920:force_original_aspect_ratio=increase:flags=bicubic,"
            "crop=1080:1920,setsar=1"
        )

    src_aspect = (src_width / src_height) if src_width and src_height else 16 / 9
    if src_aspect < 16 / 9 - 0.01:
        # вертикальный исходник → размытая подложка + видео по центру
        return (
            "split=2[bg0][fg0];"
            "[bg0]scale=1920:1080:force_original_aspect_ratio=increase:flags=bicubic,"
            "crop=1920:1080,gblur=sigma=28,eq=brightness=-0.08[bg];"
            "[fg0]scale=1080:1920:force_original_aspect_ratio=decrease:flags=bicubic[fg];"
            "[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1"
        )
    return (
        "scale=1920:1080:force_original_aspect_ratio=increase:flags=bicubic,"
        "crop=1920:1080,setsar=1"
    )


def trim_filter_graph(
    segments: Sequence[tuple[float, float]],
    *,
    audio_filter: str = "",
    video_post: str = "",
) -> tuple[str, str]:
    """Строит filter_complex для нарезки и склейки отрезков.

    Возвращает ``(filter_complex, output_map)``; ``output_map`` — метки ``[v]``/``[a]``.
    """
    if not segments:
        raise ValueError("Нет сегментов для склейки")

    parts: list[str] = []
    labels: list[str] = []
    for index, (start, end) in enumerate(segments):
        parts.append(
            f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS[v{index}]"
        )
        parts.append(
            f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[a{index}]"
        )
        labels.append(f"[v{index}][a{index}]")

    chain = f"{''.join(labels)}concat=n={len(segments)}:v=1:a=1[vout][aout]"
    if audio_filter:
        chain += f";[aout]{audio_filter}[aout2]"
        out_v, out_a = "[vout]", "[aout2]"
    else:
        out_v, out_a = "[vout]", "[aout]"
    if video_post:
        chain += f";[vout]{video_post}[vout2]"
        out_v = "[vout2]"
    parts.append(chain)
    return ";".join(parts), f"{out_v} {out_a}"
