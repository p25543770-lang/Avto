"""Веб-приложение «AI-режиссёр»: загрузка видео → авто-монтаж → ролики и шортсы."""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import re
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings
from .ffmpeg_tools import (
    cuda_available,
    ffmpeg_version,
    find_ffmpeg,
    find_ffprobe,
    gpu_name,
    nvenc_supported,
    nvenc_usable,
    resolve_encoder,
)
from .jobs import Job, JobManager
from .pipeline.llm import LLMClient
from .pipeline.runner import JobOptions, run_job
from .pipeline.transcribe import whisper_available

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger("avto.api")

WEB_DIR = Path(__file__).resolve().parent / "web"
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv"}
STATE_LOCK = threading.RLock()

settings = Settings.load()
settings.ensure_dirs()
job_manager = JobManager(max_workers=1)


# ---------------------------------------------------------------------------
# Файловый менеджер
# ---------------------------------------------------------------------------
def _safe_name(name: str) -> str:
    name = Path(name).name
    name = re.sub(r"[^\w.\- ]+", "_", name, flags=re.UNICODE).strip()
    return name or "video.mp4"


def _unique_path(directory: Path, filename: str) -> Path:
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem, suffix = Path(filename).stem, Path(filename).suffix
    index = 2
    while True:
        candidate = directory / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


# ---------------------------------------------------------------------------
# Приложение
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings.ensure_dirs()
    job_manager.ensure_running()
    watcher.start()
    LOGGER.info("Рабочие папки: %s", settings.output_path)
    try:
        yield
    finally:
        watcher.stop()
        job_manager.shutdown()
app = FastAPI(title="AI-режиссёр", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Страница и статика
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((WEB_DIR / "index.html").read_text(encoding="utf-8"))


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


# ---------------------------------------------------------------------------
# Информация о системе
# ---------------------------------------------------------------------------
@app.get("/api/system")
async def api_system() -> dict[str, Any]:
    encoder = resolve_encoder(settings.video_encoder)
    llm = LLMClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
    )
    llm_ok, llm_info = llm.ping()
    ffmpeg = find_ffmpeg()
    return {
        "ffmpeg": {
            "available": bool(ffmpeg),
            "path": ffmpeg or "",
            "version": ffmpeg_version(),
            "ffprobe": bool(find_ffprobe()),
            "nvenc": nvenc_supported(),
            "nvenc_usable": nvenc_usable(),
        },
        "gpu": {
            "name": gpu_name(),
            "cuda": cuda_available(),
        },
        "whisper": {
            "available": whisper_available(),
            "model": settings.whisper_model,
        },
        "encoder": encoder,
        "llm": {
            "available": llm_ok,
            "detail": llm_info,
            "base_url": settings.llm_base_url,
            "model": settings.llm_model,
        },
        "watch": settings.auto_watch,
        "paths": {
            "input": str(settings.input_path),
            "temp": str(settings.temp_path),
            "output": str(settings.output_path),
        },
        "platform": os.name,
    }


# ---------------------------------------------------------------------------
# Настройки
# ---------------------------------------------------------------------------
@app.get("/api/settings")
async def api_get_settings() -> dict[str, Any]:
    data = settings.to_dict()
    data["llm_api_key"] = "***" if settings.llm_api_key else ""
    return data


@app.post("/api/settings")
async def api_post_settings(payload: dict[str, Any]) -> dict[str, Any]:
    data = dict(payload or {})
    if data.get("llm_api_key") == "***":
        data.pop("llm_api_key", None)
    settings.update(data)
    settings.save()
    settings.ensure_dirs()
    return {"ok": True, "settings": api_settings_payload()}


def api_settings_payload() -> dict[str, Any]:
    data = settings.to_dict()
    data["llm_api_key"] = "***" if settings.llm_api_key else ""
    return data


# ---------------------------------------------------------------------------
# Задачи
# ---------------------------------------------------------------------------
@app.post("/api/jobs")
async def api_create_job(
    file: UploadFile = File(...),
    options: str = Form("{}"),
) -> dict[str, Any]:
    filename = _safe_name(file.filename or "video.mp4")
    if Path(filename).suffix.lower() not in VIDEO_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый формат {Path(filename).suffix}. "
            "Поддерживаются mp4, mov, mkv, avi, webm, m4v.",
        )

    try:
        job_options = JobOptions.from_dict(json.loads(options or "{}"))
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Некорректный JSON с опциями")

    settings.ensure_dirs()
    destination = _unique_path(settings.input_path, filename)

    limit = settings.max_upload_mb * 1024 * 1024
    written = 0
    try:
        with destination.open("wb") as handle:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > limit:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Файл больше {settings.max_upload_mb} МБ",
                    )
                handle.write(chunk)
    except HTTPException:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    if written == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Пустой файл")

    job = job_manager.create(destination.name, job_options.to_dict())
    _mark_processed(str(destination))
    LOGGER.info("Новая задача %s: %s", job.id, destination.name)
    _start_job(job, destination)
    return {"job_id": job.id, "source": destination.name}


def _start_job(job: Job, source: Path) -> None:
    def log(message: str) -> None:
        job_manager.log(job, message)

    def progress(value: float, message: str = "") -> None:
        job_manager.progress(job, value, message)

    def task() -> None:
        if job.cancel_requested:
            job.status = "canceled"
            job.finished_at = datetime_now()
            return
        job.status = "running"
        job.started_at = datetime_now()
        job_manager.progress(job, 2, "Подготовка")
        try:
            job.result = run_job(
                job,
                source,
                JobOptions.from_dict(job.options),
                settings,
                log=log,
                progress=progress,
            )
            job.status = "done"
            job.message = "Готово"
        except Exception as exc:  # noqa: BLE001
            if job.cancel_requested:
                job.status = "canceled"
                job.message = "Отменено"
                job_manager.log(job, "Задача отменена пользователем.")
            else:
                LOGGER.exception("Задача %s завершилась с ошибкой", job.id)
                job.status = "error"
                job.error = str(exc)
                job_manager.log(job, f"ОШИБКА: {exc}")
        finally:
            job.finished_at = datetime_now()
            job_manager.progress(job, 100 if job.status == "done" else job.progress)

    job_manager.submit(job, task)


def datetime_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


@app.get("/api/jobs")
async def api_jobs() -> dict[str, Any]:
    return {"jobs": [job.to_dict(with_logs=False) for job in job_manager.list()]}


@app.get("/api/jobs/{job_id}")
async def api_job(job_id: str) -> dict[str, Any]:
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return job.to_dict()


@app.get("/api/jobs/{job_id}/result")
async def api_job_result(job_id: str) -> dict[str, Any]:
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return {
        "status": job.status,
        "error": job.error,
        "result": job.result,
        "logs": job.logs,
    }


@app.post("/api/jobs/{job_id}/cancel")
async def api_cancel(job_id: str) -> dict[str, Any]:
    if not job_manager.cancel(job_id):
        raise HTTPException(status_code=400, detail="Задачу нельзя отменить")
    return {"ok": True}


@app.get("/api/jobs/{job_id}/events")
async def api_job_events(job_id: str, request: Request) -> StreamingResponse:
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")

    async def stream() -> AsyncIterator[str]:
        last_log_index = 0
        last_status = ""
        while True:
            if await request.is_disconnected():
                break
            current = job_manager.get(job_id)
            if current is None:
                break
            payload: dict[str, Any] = {
                "status": current.status,
                "progress": current.progress,
                "step": current.step,
                "message": current.message,
                "error": current.error,
            }
            new_logs = current.logs[last_log_index:]
            if new_logs:
                last_log_index = len(current.logs)
                payload["logs"] = new_logs
            if payload["status"] != last_status or new_logs:
                last_status = payload["status"]
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            if current.status in {"done", "error", "canceled"} and not new_logs:
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                break
            await asyncio.sleep(0.4)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Файлы
# ---------------------------------------------------------------------------
@app.get("/api/output")
async def api_output() -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(settings.output_path.iterdir()):
        if path.is_file():
            files.append(
                {
                    "name": path.name,
                    "size": path.stat().st_size,
                    "modified": path.stat().st_mtime,
                    "url": f"/api/files/{path.name}",
                    "kind": "video" if path.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"} else "file",
                }
            )
    files.sort(key=lambda item: item["modified"], reverse=True)
    return {"files": files}


@app.get("/api/files/{name}")
async def api_file(name: str, request: Request) -> Response:
    safe = _safe_name(name)
    path = settings.output_path / safe
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Файл не найден")

    file_size = path.stat().st_size
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    range_header = request.headers.get("range")

    if range_header:
        match = re.match(r"bytes=(\d*)-(\d*)", range_header)
        if match:
            start = int(match.group(1)) if match.group(1) else 0
            end = int(match.group(2)) if match.group(2) else file_size - 1
            end = min(end, file_size - 1)
            if start > end:
                return Response(
                    status_code=416,
                    headers={"Content-Range": f"bytes */{file_size}"},
                )
            length = end - start + 1

            def iterator():
                with path.open("rb") as handle:
                    handle.seek(start)
                    remaining = length
                    while remaining > 0:
                        chunk = handle.read(min(1024 * 512, remaining))
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        yield chunk

            return StreamingResponse(
                iterator(),
                status_code=206,
                media_type=media_type,
                headers={
                    "Content-Range": f"bytes {start}-{end}/{file_size}",
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(length),
                },
            )

    def full_iterator():
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 512)
                if not chunk:
                    break
                yield chunk

    return StreamingResponse(
        full_iterator(),
        media_type=media_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Content-Disposition": f'inline; filename="{safe}"',
        },
    )


# ---------------------------------------------------------------------------
# Слежение за папкой input/
# ---------------------------------------------------------------------------
class InputWatcher:
    """Следит за папкой input/ и автоматически запускает монтаж новых файлов."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._seen: dict[str, float] = {}

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> None:
        if self.is_running():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="avto-watcher", daemon=True)
        self._thread.start()
        LOGGER.info("Слежение за папкой input/ включено")

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                if settings.auto_watch:
                    self._scan()
            except Exception:  # noqa: BLE001
                LOGGER.exception("Ошибка сканирования папки input")
            self._stop.wait(8.0)

    def _scan(self) -> None:
        directory = settings.input_path
        if not directory.exists():
            return
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            stat = path.stat()
            key = str(path)
            previous = self._seen.get(key)
            self._seen[key] = stat.st_mtime
            if previous is None:
                self._seen[key] = stat.st_mtime
                continue
            # файл изменился (докачивается) — ждём стабилизации
            if abs(previous - stat.st_mtime) > 0.5:
                continue
            if _already_processed(key):
                continue
            _mark_processed(key)
            LOGGER.info("Слежение: новый файл %s", path.name)
            job = job_manager.create(path.name, JobOptions().to_dict())
            _start_job(job, path)

    def forget(self, path: str) -> None:
        self._seen.pop(path, None)


watcher = InputWatcher()
_processed_files: set[str] = set()


def _already_processed(key: str) -> bool:
    with STATE_LOCK:
        return key in _processed_files


def _mark_processed(key: str) -> None:
    with STATE_LOCK:
        _processed_files.add(key)


@app.post("/api/watch")
async def api_watch(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    enabled = bool((payload or {}).get("enabled", True))
    settings.auto_watch = enabled
    settings.save()
    if enabled:
        watcher.start()
    return {"ok": True, "enabled": enabled}


@app.get("/api/watch")
async def api_watch_state() -> dict[str, Any]:
    return {"enabled": settings.auto_watch, "running": watcher.is_running()}
