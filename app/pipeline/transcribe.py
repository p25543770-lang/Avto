"""Распознавание речи через faster-whisper с контролем видеопамяти.

Правила (критичны для 8 ГБ VRAM):
* модель грузится только на этапе транскрибации;
* сразу после получения текста объект модели удаляется, вызывается
  ``gc.collect()`` и ``torch.cuda.empty_cache()``;
* тяжёлый рендер никогда не запускается параллельно с распознаванием.
"""

from __future__ import annotations

import gc
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

LOGGER = logging.getLogger("avto.transcribe")

LogFn = Callable[[str], None]


class TranscriptionError(RuntimeError):
    """Ошибка распознавания речи."""


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "words": [
                {"start": round(w.start, 3), "end": round(w.end, 3), "text": w.text}
                for w in self.words
            ],
        }


@dataclass
class Transcript:
    language: str = "ru"
    duration: float = 0.0
    segments: list[TranscriptSegment] = field(default_factory=list)
    backend: str = "faster-whisper"

    @property
    def text(self) -> str:
        return " ".join(seg.text.strip() for seg in self.segments if seg.text.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "duration": round(self.duration, 3),
            "backend": self.backend,
            "segments": [seg.to_dict() for seg in self.segments],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "Transcript":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        segments = [
            TranscriptSegment(
                start=float(seg.get("start", 0.0)),
                end=float(seg.get("end", 0.0)),
                text=str(seg.get("text", "")),
                words=[
                    Word(
                        start=float(w.get("start", 0.0)),
                        end=float(w.get("end", 0.0)),
                        text=str(w.get("text", "")),
                    )
                    for w in seg.get("words", [])
                ],
            )
            for seg in data.get("segments", [])
        ]
        return cls(
            language=data.get("language", "ru"),
            duration=float(data.get("duration", 0.0)),
            segments=segments,
            backend=data.get("backend", "faster-whisper"),
        )

    def plain_text(self) -> str:
        """Текст с таймкодами — то, что уходит в LLM."""
        lines = []
        for seg in self.segments:
            if not seg.text.strip():
                continue
            lines.append(
                f"[{_fmt_time(seg.start)} - {_fmt_time(seg.end)}] {seg.text.strip()}"
            )
        return "\n".join(lines)


def _fmt_time(value: float) -> str:
    value = max(0.0, float(value))
    minutes = int(value // 60)
    seconds = value - minutes * 60
    return f"{minutes:02d}:{seconds:05.2f}"


# ---------------------------------------------------------------------------
# Загрузка модели
# ---------------------------------------------------------------------------
def _resolve_device(requested: str) -> str:
    if requested and requested != "auto":
        return requested
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # pragma: no cover
        pass
    return "cpu"


def _resolve_compute_type(requested: str, device: str) -> str:
    if requested and requested != "auto":
        return requested
    return "int8_float16" if device == "cuda" else "int8"


def release_gpu() -> None:
    """Принудительно освобождает видеопамять."""
    gc.collect()
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:  # pragma: no cover
        pass


def whisper_available() -> bool:
    try:
        import faster_whisper  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Основная функция
# ---------------------------------------------------------------------------
def transcribe(
    audio_path: Path | str,
    *,
    model_name: str = "medium",
    device: str = "auto",
    compute_type: str = "auto",
    language: str = "ru",
    backend: str = "auto",
    log: LogFn | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> Transcript:
    """Распознаёт речь и возвращает транскрипт с таймкодами."""
    emit = log or (lambda message: LOGGER.info(message))
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise TranscriptionError(f"Аудиофайл не найден: {audio_path}")

    if backend == "stub":
        emit("Режим диагностики: транскрипция без нейросети (на основе пауз).")
        return _stub_transcript(audio_path, language)

    if not whisper_available():
        raise TranscriptionError(
            "faster-whisper не установлен. Выполните: pip install faster-whisper "
            "(в Windows также нужен Visual C++ Redistributable)"
        )

    resolved_device = _resolve_device(device)
    resolved_compute = _resolve_compute_type(compute_type, resolved_device)
    if resolved_device == "cpu":
        emit(
            "CUDA недоступна — распознавание на CPU будет медленным. "
            "Проверьте установку драйвера NVIDIA и torch с CUDA."
        )
    emit(
        f"Модель: {model_name}, device={resolved_device}, "
        f"compute_type={resolved_compute}"
    )

    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise TranscriptionError(f"Не удалось импортировать faster-whisper: {exc}") from exc

    model = None
    try:
        model = WhisperModel(
            model_name,
            device=resolved_device,
            compute_type=resolved_compute,
        )
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=(language or None),
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400},
            beam_size=5,
        )
        segments: list[TranscriptSegment] = []
        for segment in segments_iter:
            if should_stop is not None and should_stop():
                raise TranscriptionError("Операция отменена пользователем")
            words = [
                Word(start=float(w.start), end=float(w.end), text=str(w.text))
                for w in (getattr(segment, "words", None) or [])
            ]
            segments.append(
                TranscriptSegment(
                    start=float(segment.start),
                    end=float(segment.end),
                    text=str(segment.text or "").strip(),
                    words=words,
                )
            )
            emit(f"  распознано: {segment.start:7.2f}s  {segment.text.strip()[:70]}")
        transcript = Transcript(
            language=getattr(info, "language", language) or language,
            duration=float(getattr(info, "duration", 0.0) or 0.0),
            segments=segments,
        )
        emit(
            f"Транскрипция готова: {len(segments)} сегментов, "
            f"длительность {transcript.duration:.1f} с"
        )
        return transcript
    except TranscriptionError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise TranscriptionError(f"Ошибка распознавания: {exc}") from exc
    finally:
        # строгое освобождение VRAM: модель могла не вернуться из except
        try:
            del model
        except Exception:  # pragma: no cover
            pass
        release_gpu()
        emit("Видеопамять освобождена.")


# ---------------------------------------------------------------------------
# Диагностический backend без нейросети
# ---------------------------------------------------------------------------
def _stub_transcript(audio_path: Path, language: str) -> Transcript:
    """Сегментация по тишине — чтобы проверить монтаж без загрузки моделей."""
    from ..ffmpeg_tools import MediaError, run_cmd, require_ffmpeg

    try:
        proc = run_cmd(
            [
                require_ffmpeg(),
                "-hide_banner",
                "-nostdin",
                "-i",
                str(audio_path),
                "-af",
                "silencedetect=noise=-32dB:d=0.4",
                "-f",
                "null",
                "-",
            ],
            check=False,
        )
    except MediaError:
        return Transcript(language=language, backend="stub")

    text = (proc.stderr or "") + (proc.stdout or "")
    silences: list[tuple[float, float]] = []
    for match in re.finditer(
        r"silence_start:\s*(-?[\d.]+)\s*\|.*?silence_end:\s*([\d.]+)", text
    ):
        silences.append((float(match.group(1)), float(match.group(2))))

    duration = 0.0
    dur_match = re.search(r"time=(\d+):(\d+):([\d.]+)", text)
    if dur_match:
        duration = (
            int(dur_match.group(1)) * 3600
            + int(dur_match.group(2)) * 60
            + float(dur_match.group(3))
        )

    speech: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in silences:
        if start > cursor:
            speech.append((cursor, start))
        cursor = max(cursor, end)
    if duration > cursor:
        speech.append((cursor, duration))

    segments = [
        TranscriptSegment(start=s, end=e, text=f"[речь {s:.1f}-{e:.1f}s]")
        for s, e in speech
        if e - s > 0.4
    ]
    return Transcript(
        language=language, duration=duration, segments=segments, backend="stub"
    )
