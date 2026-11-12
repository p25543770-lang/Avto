"""Тесты работы с ffmpeg: фильтры, кодировщики, probe."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ffmpeg_tools import (
    aspect_filter,
    encoder_args,
    find_ffmpeg,
    loudnorm_filter,
    probe,
    require_ffmpeg,
    resolve_encoder,
    trim_filter_graph,
)


def test_aspect_filter_9x16_crops_center():
    chain = aspect_filter("9:16", 1920, 1080)
    assert "crop=1080:1920" in chain
    assert "scale=1080:1920" in chain


def test_aspect_filter_16x9_from_vertical_uses_blur_background():
    chain = aspect_filter("16:9", 1080, 1920)
    assert "overlay" in chain
    assert "gblur" in chain


def test_aspect_filter_16x9_from_wide_crops():
    chain = aspect_filter("16:9", 2560, 1440)
    assert "crop=1920:1080" in chain
    assert "overlay" not in chain


def test_loudnorm_matches_spec():
    assert loudnorm_filter(-14, -1.5, 11) == "loudnorm=I=-14:TP=-1.5:LRA=11"


def test_encoder_args_nvenc():
    args = encoder_args("h264_nvenc", nvenc_preset="p4", nvenc_cq=20)
    assert args[:2] == ["-c:v", "h264_nvenc"]
    assert "-preset" in args and "p4" in args
    assert "20" in args  # cq
    assert args[-3:] == ["-c:a", "aac", "-b:a"] or "-b:a" in args


def test_encoder_args_libx264_has_crf_not_cq():
    args = encoder_args("libx264", nvenc_cq=20)
    assert "-crf" in args
    assert "-preset" in args


def test_resolve_encoder_prefers_requested():
    assert resolve_encoder("libx264") == "libx264"
    assert resolve_encoder("h264_nvenc") == "h264_nvenc"


def test_trim_filter_graph_structure():
    graph, mapping = trim_filter_graph([(0.0, 5.0), (10.0, 15.0)], audio_filter="loudnorm")
    assert graph.count("[0:v]trim=") == 2
    assert graph.count("[0:a]atrim=") == 2
    assert "atrim=start=10.000:end=15.000" in graph
    assert "concat=n=2:v=1:a=1" in graph
    assert "loudnorm" in graph
    assert mapping.count("[") == 2


def test_trim_filter_graph_requires_segments():
    with pytest.raises(ValueError):
        trim_filter_graph([])


@pytest.mark.skipif(not find_ffmpeg(), reason="ffmpeg недоступен")
def test_probe_synthetic_video(tmp_path: Path):
    import subprocess

    target = tmp_path / "probe.mp4"
    subprocess.run(
        [
            require_ffmpeg(),
            "-y",
            "-hide_banner",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=640x360:rate=25:duration=3",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=3",
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
    info = probe(target)
    assert info.width == 640
    assert info.height == 360
    assert 2.5 <= info.duration <= 4.0
    assert info.has_audio
