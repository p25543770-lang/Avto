"""Тесты логики смысловой редактуры: таймкоды, дубли, главы, шортсы."""

from __future__ import annotations

from app.pipeline.edit import (
    Interval,
    ShortCandidate,
    chunk_transcript,
    complement_intervals,
    format_timestamp,
    heuristic_keep_segments,
    heuristic_shorts,
    merge_intervals,
    parse_timestamp,
    similarity,
    _enforce_short_rules,
)
from app.pipeline.transcribe import Transcript, TranscriptSegment


def _transcript(pairs):
    segments = [
        TranscriptSegment(start=start, end=end, text=text) for start, end, text in pairs
    ]
    return Transcript(language="ru", duration=segments[-1].end if segments else 0, segments=segments)


def test_format_timestamp():
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(65) == "01:05"
    assert format_timestamp(3599.6) == "60:00"  # округление до минуты


def test_parse_timestamp_variants():
    assert parse_timestamp("01:30") == 90
    assert parse_timestamp("00:01:30") == 90
    assert parse_timestamp("12.5") == 12.5
    assert parse_timestamp(7) == 7


def test_merge_intervals_joins_close_ranges():
    merged = merge_intervals([Interval(0, 5), Interval(5.2, 10), Interval(20, 25)])
    assert len(merged) == 2
    assert merged[0].end == 10


def test_complement_intervals():
    gaps = complement_intervals([Interval(10, 20)], duration=30)
    assert [(round(g.start, 2), round(g.end, 2)) for g in gaps] == [(0, 10), (20, 30)]


def test_similarity():
    assert similarity(["привет", "как", "дела"], ["привет", "как", "дела"]) == 1.0
    assert similarity(["а"], ["б"]) == 0.0


def test_heuristic_removes_duplicate_earlier_phrase():
    transcript = _transcript(
        [
            (0.0, 4.0, "Сегодня мы поговорим о монтаже видео"),
            (4.5, 8.5, "Сегодня мы поговорим о монтаже видео целиком"),
            (9.0, 14.0, "Первая часть ролика про автоматический монтаж"),
        ]
    )
    keep = heuristic_keep_segments(transcript, transcript.duration)
    total = sum(item.duration for item in keep)
    # первый (дублирующий) фрагмент должен быть вырезан
    assert total < transcript.duration
    covered_first = any(item.start <= 0.1 and item.end >= 3.9 for item in keep)
    assert not covered_first


def test_heuristic_keeps_unique_speech():
    transcript = _transcript(
        [
            (0.0, 4.0, "Привет, это первая часть"),
            (5.0, 9.0, "А теперь вторая часть совсем про другое"),
        ]
    )
    keep = heuristic_keep_segments(transcript, transcript.duration)
    assert sum(item.duration for item in keep) > transcript.duration - 1.0


def test_chunk_transcript_splits_by_time():
    segments = [
        TranscriptSegment(start=i * 100.0, end=i * 100.0 + 50.0, text=f"Фрагмент {i}")
        for i in range(12)
    ]
    transcript = Transcript(segments=segments, duration=1200)
    chunks = chunk_transcript(transcript, max_chars=10_000, max_seconds=400)
    assert len(chunks) >= 2
    assert chunks[0].offset == 0.0
    assert chunks[1].offset > 0


def test_enforce_short_rules_respects_limits_and_overlap():
    candidates = [
        ShortCandidate(start=0, end=40, score=9),
        ShortCandidate(start=5, end=45, score=8),
        ShortCandidate(start=200, end=230, score=7),
    ]
    picked = _enforce_short_rules(candidates, duration=300, count=3, min_duration=30, max_duration=55)
    assert len(picked) == 2  # второй пересекается с первым
    assert all(30 <= item.duration <= 55 for item in picked)
    assert picked[0].end - picked[0].start <= 55


def test_enforce_short_rules_truncates_long_candidate():
    picked = _enforce_short_rules(
        [ShortCandidate(start=0, end=300)], duration=300, count=1, min_duration=30, max_duration=55
    )
    assert picked[0].duration == 55


def test_heuristic_shorts_picks_spread_windows():
    segments = [
        TranscriptSegment(start=i * 5.0, end=i * 5.0 + 4.0, text="слово " * 8)
        for i in range(20)
    ]
    transcript = Transcript(segments=segments, duration=100)
    shorts = heuristic_shorts(transcript, 100, 2, 20, 40)
    assert len(shorts) <= 2
    for item in shorts:
        assert 20 <= item.duration <= 40
