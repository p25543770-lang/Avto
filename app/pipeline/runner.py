"""Оркестрация полного цикла монтажа: от исходника до готовых роликов."""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

from ..config import Settings
from ..ffmpeg_tools import probe
from ..jobs import Job
from .edit import Interval, plan_edit
from .llm import LLMClient
from .render import render_final, render_short
from .silence import remove_silence
from .transcribe import TranscriptionError, Transcript, transcribe

LOGGER = logging.getLogger("avto.pipeline")

LogFn = Callable[[str], None]
ProgressFn = Callable[[float, str], None]

ASPECT_SLUGS = {"16:9": "16x9", "9:16": "9x16"}


@dataclass
class JobOptions:
    """Настройки конкретной задачи (приходят из веб-интерфейса)."""

    final_formats: list[str] = field(default_factory=lambda: ["16:9", "9:16"])
    short_formats: list[str] = field(default_factory=lambda: ["9:16"])
    short_count: int = 3
    short_min_duration: float = 30.0
    short_max_duration: float = 55.0
    silence_margin: float = 0.2
    language: str = "ru"
    whisper_model: str = "medium"
    remove_duplicates: bool = True
    encoder: str = "auto"
    use_llm: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "JobOptions":
        payload = payload or {}
        options = cls()

        def _formats(key: str, default: list[str]) -> list[str]:
            raw = payload.get(key)
            if raw is None:
                return default
            if isinstance(raw, str):
                raw = [raw]
            valid = [str(item) for item in raw if str(item) in ASPECT_SLUGS]
            return valid or default

        options.final_formats = _formats("final_formats", options.final_formats)
        options.short_formats = _formats("short_formats", options.short_formats)
        try:
            count = int(payload.get("short_count", 3))
        except (TypeError, ValueError):
            count = 3
        options.short_count = max(1, min(5, count))
        min_duration = float(payload.get("short_min_duration", 30) or 30)
        max_duration = float(payload.get("short_max_duration", 55) or 55)
        options.short_min_duration = max(5.0, min(120.0, min_duration))
        options.short_max_duration = max(
            options.short_min_duration, min(180.0, max_duration)
        )
        options.silence_margin = max(
            0.0, min(3.0, float(payload.get("silence_margin", 0.2) or 0.0))
        )
        options.language = str(payload.get("language", "ru") or "ru")
        options.whisper_model = str(payload.get("whisper_model", "medium") or "medium")
        options.remove_duplicates = bool(payload.get("remove_duplicates", True))
        options.encoder = str(payload.get("encoder", "auto") or "auto")
        options.use_llm = bool(payload.get("use_llm", True))
        return options

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StepResult:
    name: str
    status: str
    detail: str = ""


class JobCancelled(RuntimeError):
    """Задача остановлена пользователем."""


def _slugify(name: str) -> str:
    stem = Path(name).stem
    cleaned = "".join(
        char if (char.isalnum() or char in "-_") else "_" for char in stem
    ).strip("_")
    return cleaned or "video"


def run_job(
    job: Job,
    source: Path,
    options: JobOptions,
    settings: Settings,
    *,
    log: LogFn,
    progress: ProgressFn,
) -> dict[str, Any]:
    """Выполняет полный цикл и возвращает описание результатов."""
    started = time.time()
    settings.ensure_dirs()
    stem = _slugify(source.name)
    work = settings.temp_path / f"{stem}_{job.id}"
    work.mkdir(parents=True, exist_ok=True)

    clean_path = work / "clean.mp4"
    audio_path = work / "audio.wav"
    transcript_path = work / "transcript.json"
    plan_path = work / "edit.json"

    outputs: list[dict[str, Any]] = []
    shorts_out: list[dict[str, Any]] = []
    chapters: list[dict[str, Any]] = []

    def should_stop() -> bool:
        return bool(job.cancel_requested)

    def check_cancelled() -> None:
        if should_stop():
            raise JobCancelled("Задача отменена пользователем")

    try:
        # --- Шаг 1: удаление тишины ---------------------------------
        progress(5, "Удаление пауз")
        log(f"[1/6] Удаляю паузы из {source.name} (отступ {options.silence_margin}s)")
        info = probe(source)
        log(
            f"      Исходник: {info.width}x{info.height}, {info.duration:.1f} с, "
            f"{'со звуком' if info.has_audio else 'без звука'}"
        )
        remove_silence(
            source,
            clean_path,
            margin=options.silence_margin,
            settings=settings,
            log=log,
            should_stop=should_stop,
        )
        check_cancelled()
        clean_info = probe(clean_path)
        removed = max(0.0, info.duration - clean_info.duration)
        log(
            f"      Готово: {clean_info.duration:.1f} с "
            f"(удалено {removed:.1f} с тишины)"
        )

        # --- Шаг 2: транскрибация -----------------------------------
        progress(20, "Распознавание речи")
        log("[2/6] Распознаю речь (faster-whisper)…")
        if clean_info.has_audio:
            _extract_audio(clean_path, audio_path, log)
            transcript = transcribe(
                audio_path,
                model_name=options.whisper_model,
                device=settings.whisper_device,
                compute_type=settings.whisper_compute_type,
                language=options.language,
                backend=settings.transcribe_backend,
                log=log,
                should_stop=should_stop,
            )
            transcript.duration = clean_info.duration
            transcript.save(transcript_path)
            log(f"      Слов: {len(transcript.text.split())}")
        else:
            transcript = Transcript(language=options.language, duration=clean_info.duration)
            log("      В исходнике нет аудио — шаг распознавания пропущен.")

        # --- Шаг 3: смысловая редактура ----------------------------
        progress(38, "Смысловая редактура")
        log("[3/6] Анализирую смысл: ищу дубли, ставлю главы, выбираю шортсы…")
        llm = None
        if options.use_llm:
            llm = LLMClient(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
                model=settings.llm_model,
                timeout=settings.llm_timeout,
                max_tokens=settings.llm_max_tokens,
            )
        plan = plan_edit(
            transcript,
            llm=llm if options.remove_duplicates or options.short_count else None,
            log=log,
            short_count=options.short_count,
            short_min_duration=options.short_min_duration,
            short_max_duration=options.short_max_duration,
        )
        if not options.remove_duplicates:
            plan.keep = [Interval(0.0, clean_info.duration)]
            plan.source = "без удаления дублей"
        plan_path.write_text(
            json.dumps(plan.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        kept = sum(i.duration for i in plan.keep)
        log(
            f"      План монтажа ({plan.source}): оставлено {kept:.1f} с, "
            f"глав — {len(plan.chapters)}, шортсов — {len(plan.shorts)}"
        )

        # --- Шаг 4: чистовик ----------------------------------------
        progress(50, "Рендер чистовика")
        log(f"[4/6] Рендер чистового ролика в форматах: {', '.join(options.final_formats)}")
        multi = len(options.final_formats) > 1
        for index, aspect in enumerate(options.final_formats):
            check_cancelled()
            suffix = f"_{ASPECT_SLUGS[aspect]}" if multi else ""
            destination = settings.output_path / f"{stem}_final{suffix}.mp4"
            render_final(
                clean_path,
                plan.keep,
                destination,
                aspect,
                settings,
                log=log,
                should_stop=should_stop,
            )
            final_info = probe(destination)
            outputs.append(
                {
                    "kind": "final",
                    "aspect": aspect,
                    "filename": destination.name,
                    "path": str(destination),
                    "duration": round(final_info.duration, 2),
                    "width": final_info.width,
                    "height": final_info.height,
                    "size": destination.stat().st_size,
                }
            )
            progress(50 + 12 * (index + 1) / max(1, len(options.final_formats)),
                     f"Чистовик {aspect} готов")

        # --- Шаг 5: шортсы ------------------------------------------
        progress(65, "Рендер шортсов")
        log(f"[5/6] Рендер шортсов ({len(plan.shorts)} шт.) в форматах: "
            f"{', '.join(options.short_formats)}")
        short_multi = len(options.short_formats) > 1
        for index, short in enumerate(plan.shorts):
            for aspect in options.short_formats:
                check_cancelled()
                suffix = f"_{ASPECT_SLUGS[aspect]}" if short_multi else ""
                destination = (
                    settings.output_path / f"{stem}_short_{index + 1}{suffix}.mp4"
                )
                render_short(
                    clean_path,
                    Interval(short.start, short.end),
                    destination,
                    aspect,
                    settings,
                    log=log,
                    should_stop=should_stop,
                )
                short_info = probe(destination)
                shorts_out.append(
                    {
                        "kind": "short",
                        "index": index + 1,
                        "aspect": aspect,
                        "title": short.title or f"Шортс #{index + 1}",
                        "hook": short.hook,
                        "start": round(short.start, 2),
                        "end": round(short.end, 2),
                        "filename": destination.name,
                        "path": str(destination),
                        "duration": round(short_info.duration, 2),
                        "width": short_info.width,
                        "height": short_info.height,
                        "size": destination.stat().st_size,
                    }
                )
            progress(
                65 + 20 * (index + 1) / max(1, len(plan.shorts)),
                f"Шортс {index + 1}/{len(plan.shorts)} готов",
            )

        # --- Шаг 6: главы и уборка ----------------------------------
        progress(90, "Сохранение таймкодов глав")
        log("[6/6] Сохраняю таймкоды глав…")
        chapters = [chapter.to_dict() for chapter in plan.chapters]
        chapters_file = settings.output_path / f"{stem}_chapters.txt"
        lines = [f"# Главы: {source.name}", ""]
        for chapter in chapters:
            lines.append(f"{chapter['time']}  {chapter['title']}")
        if not lines[-1]:
            lines.append("(главы не определены)")
        chapters_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        transcript_file = settings.output_path / f"{stem}_transcript.txt"
        transcript_file.write_text(transcript.plain_text() + "\n", encoding="utf-8")

        result = {
            "source": source.name,
            "stem": stem,
            "final": outputs,
            "shorts": shorts_out,
            "chapters": chapters,
            "chapters_file": str(chapters_file),
            "chapters_filename": chapters_file.name,
            "transcript_filename": transcript_file.name,
            "transcript_file": str(transcript_file),
            "plan": plan.to_dict(),
            "duration_source": round(info.duration, 2),
            "duration_final": round(clean_info.duration, 2),
            "elapsed": round(time.time() - started, 1),
            "encoder": settings.video_encoder,
        }
        progress(100, "Готово")
        log(f"Готово за {result['elapsed']} с. Файлы в {settings.output_path}")
        return result
    finally:
        if not settings.keep_temp:
            shutil.rmtree(work, ignore_errors=True)
        else:
            log(f"Временные файлы сохранены: {work}")


def _extract_audio(source: Path, destination: Path, log: LogFn) -> None:
    """Извлекает моно-звук 16 кГц для faster-whisper."""
    from ..ffmpeg_tools import MediaError, require_ffmpeg, run_cmd

    cmd = [
        require_ffmpeg(),
        "-y",
        "-hide_banner",
        "-nostdin",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]
    try:
        run_cmd(cmd)
    except MediaError as exc:
        raise TranscriptionError(f"Не удалось извлечь аудио: {exc}") from exc
