"""Тесты опций задачи и планировщика монтажа."""

from __future__ import annotations

from app.pipeline.edit import plan_edit
from app.pipeline.runner import JobOptions
from app.pipeline.transcribe import Transcript, TranscriptSegment


def test_options_defaults():
    options = JobOptions.from_dict({})
    assert options.final_formats == ["16:9", "9:16"]
    assert options.short_formats == ["9:16"]
    assert options.short_count == 3
    assert options.silence_margin == 0.2


def test_options_filter_invalid_aspects():
    options = JobOptions.from_dict(
        {"final_formats": ["16:9", "4:3", "21:9"], "short_formats": ["9:16", "square"]}
    )
    assert options.final_formats == ["16:9"]
    assert options.short_formats == ["9:16"]


def test_options_min_duration_never_exceeds_max():
    options = JobOptions.from_dict({"short_min_duration": 90, "short_max_duration": 10})
    assert options.short_min_duration <= options.short_max_duration
    options = JobOptions.from_dict({"short_min_duration": 3, "short_max_duration": 8})
    assert options.short_min_duration <= options.short_max_duration


def test_options_short_count_clamped():
    assert JobOptions.from_dict({"short_count": 0}).short_count == 1
    assert JobOptions.from_dict({"short_count": 99}).short_count == 5


class FakeLLM:
    """Заглушка LLM: возвращает заранее заданный JSON."""

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def chat_json(self, system, user):  # noqa: D102
        self.calls += 1
        return self.payload


def _transcript():
    segments = [
        TranscriptSegment(start=0.0, end=20.0, text="Первая мысль ролика"),
        TranscriptSegment(start=20.5, end=50.0, text="Вторая мысль ролика"),
        TranscriptSegment(start=50.5, end=90.0, text="Третья мысль ролика"),
        TranscriptSegment(start=90.5, end=120.0, text="Четвёртая мысль ролика"),
    ]
    return Transcript(language="ru", duration=120.0, segments=segments)


def test_plan_edit_uses_llm_keep_segments():
    llm = FakeLLM(
        {
            "keep_segments": [{"start": 0.0, "end": 90.0}],
            "chapters": [{"time": "00:00", "title": "Начало"}],
            "moments": [
                {"start": 30.0, "end": 75.0, "title": "Момент", "hook": "Хук", "score": 8}
            ],
        }
    )
    plan = plan_edit(_transcript(), llm=llm, log=lambda _: None, short_count=1,
                     short_min_duration=30, short_max_duration=55)
    assert plan.source == "llm"
    assert [(i.start, i.end) for i in plan.keep] == [(0.0, 90.0)]
    assert plan.chapters and plan.chapters[0].title == "Начало"
    assert len(plan.shorts) == 1
    assert 30 <= plan.shorts[0].duration <= 55


def test_plan_edit_falls_back_when_llm_returns_nothing():
    llm = FakeLLM({})
    plan = plan_edit(_transcript(), llm=llm, log=lambda _: None, short_count=2,
                     short_min_duration=5, short_max_duration=10)
    assert plan.keep  # ролик не потерян
    assert plan.source in {"llm", "heuristic"}


def test_plan_edit_without_llm_uses_heuristics():
    plan = plan_edit(_transcript(), llm=None, log=lambda _: None, short_count=2,
                     short_min_duration=5, short_max_duration=10)
    assert plan.source == "heuristic"
    assert plan.keep
