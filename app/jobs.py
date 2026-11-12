"""Менеджер задач: очередь, прогресс, логи, отмена.

Все задачи выполняются последовательно (один воркер) — это гарантирует, что
faster-whisper никогда не запустится одновременно с тяжёлым рендером, а 8 ГБ
VRAM не исчерпаются.
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable

JobCallback = Callable[["Job", str], None]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    id: str
    source_name: str
    options: dict[str, Any] = field(default_factory=dict)
    status: str = "queued"  # queued | running | done | error | canceled
    step: str = ""
    progress: float = 0.0
    message: str = ""
    error: str = ""
    created_at: str = field(default_factory=_now)
    started_at: str = ""
    finished_at: str = ""
    logs: list[dict[str, str]] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)
    cancel_requested: bool = False

    def to_dict(self, with_logs: bool = True) -> dict[str, Any]:
        data = asdict(self)
        data.pop("cancel_requested", None)
        if not with_logs:
            data.pop("logs", None)
        return data


class JobManager:
    """Очередь задач с одним воркером."""

    def __init__(self, max_workers: int = 1) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()
        self._max_workers = max_workers
        self._executor: ThreadPoolExecutor | None = None
        self._order: list[str] = []

    def ensure_running(self) -> None:
        """Создаёт (или пересоздаёт) пул потоков — после shutdown он не нужен."""
        with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=self._max_workers, thread_name_prefix="avto-job"
                )

    # ------------------------------------------------------------------
    def create(
        self, source_name: str, options: dict[str, Any] | None = None
    ) -> Job:
        job = Job(
            id=uuid.uuid4().hex[:12],
            source_name=source_name,
            options=options or {},
        )
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            # храним историю последних 50 задач
            while len(self._order) > 50:
                stale = self._order.pop(0)
                self._jobs.pop(stale, None)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return [self._jobs[jid] for jid in reversed(self._order)]

    def submit(self, job: Job, func: Callable[[], None]) -> None:
        def wrapper() -> None:
            try:
                func()
            except Exception as exc:  # noqa: BLE001
                job.status = "error"
                job.error = str(exc)
                job.finished_at = _now()
                self.log(job, f"ОШИБКА: {exc}")

        self.ensure_running()
        assert self._executor is not None
        self._executor.submit(wrapper)

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None:
            return False
        if job.status in {"queued", "running"}:
            job.cancel_requested = True
            job.message = "Остановка по запросу..."
            self.log(job, "Задача остановлена по запросу пользователя.")
            if job.status == "queued":
                job.status = "canceled"
                job.finished_at = _now()
            return True
        return False

    # ------------------------------------------------------------------
    @staticmethod
    def log(job: Job, message: str) -> None:
        job.logs.append({"time": datetime.now().strftime("%H:%M:%S"), "text": message})
        if len(job.logs) > 800:
            del job.logs[:-800]

    @staticmethod
    def progress(job: Job, value: float, message: str = "") -> None:
        job.progress = max(0.0, min(100.0, float(value)))
        if message:
            job.message = message
            job.step = message

    def shutdown(self) -> None:
        with self._lock:
            if self._executor is not None:
                self._executor.shutdown(wait=False, cancel_futures=True)
                self._executor = None
