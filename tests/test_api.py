"""Интеграционный тест веб-приложения: загрузка → монтаж → готовые файлы."""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.ffmpeg_tools import find_ffmpeg, require_ffmpeg
from app.main import app, settings

from .stub_llm import Handler

pytestmark = pytest.mark.skipif(not find_ffmpeg(), reason="ffmpeg недоступен")


@pytest.fixture(scope="module")
def stub_llm():
    from http.server import HTTPServer

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory):
    target = tmp_path_factory.mktemp("media") / "sample.mp4"
    subprocess.run(
        [
            require_ffmpeg(),
            "-y",
            "-hide_banner",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=640x360:rate=25:duration=8",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=330:duration=8",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(target),
        ],
        check=True,
        capture_output=True,
    )
    return target


def test_system_endpoint():
    with TestClient(app) as client:
        data = client.get("/api/system").json()
    assert "ffmpeg" in data
    assert "gpu" in data
    assert data["encoder"] in {"h264_nvenc", "libx264"}


def test_index_page_served():
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "AI-режиссёр" in response.text


def test_full_pipeline(stub_llm, sample_video, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "transcribe_backend", "stub")
    monkeypatch.setattr(settings, "llm_base_url", stub_llm)
    monkeypatch.setattr(settings, "llm_model", "stub-model")
    monkeypatch.setattr(settings, "input_dir", str(tmp_path / "input"))
    monkeypatch.setattr(settings, "temp_dir", str(tmp_path / "temp"))
    monkeypatch.setattr(settings, "output_dir", str(tmp_path / "output"))
    settings.ensure_dirs()

    options = {
        "final_formats": ["16:9", "9:16"],
        "short_formats": ["9:16"],
        "short_count": 1,
        "short_min_duration": 3,
        "short_max_duration": 8,
    }

    with TestClient(app) as client:
        with sample_video.open("rb") as handle:
            response = client.post(
                "/api/jobs",
                files={"file": ("sample.mp4", handle, "video/mp4")},
                data={"options": __import__("json").dumps(options)},
            )
        assert response.status_code == 200, response.text
        job_id = response.json()["job_id"]

        deadline = time.time() + 240
        status = "queued"
        while time.time() < deadline:
            payload = client.get(f"/api/jobs/{job_id}").json()
            status = payload["status"]
            if status in {"done", "error", "canceled"}:
                break
            time.sleep(1.0)

        assert status == "done", payload.get("error")

        result = client.get(f"/api/jobs/{job_id}/result").json()["result"]
        assert len(result["final"]) == 2
        aspects = {item["aspect"] for item in result["final"]}
        assert aspects == {"16:9", "9:16"}
        assert result["shorts"], "ожидался хотя бы один шортс"

        for item in result["final"] + result["shorts"]:
            path = Path(item["path"])
            assert path.exists() and path.stat().st_size > 0
            # файл должен отдаваться по HTTP
            served = client.get(f"/api/files/{item['filename']}")
            assert served.status_code == 200
            assert served.headers["content-type"] == "video/mp4"

        chapters_file = Path(result["chapters_file"])
        assert chapters_file.exists()
