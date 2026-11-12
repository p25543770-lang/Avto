"""Смысловая редактура: поиск дублей, главы и выбор шортсов.

Основной режим — локальный LLM (OpenAI-совместимый). Если сервер недоступен,
работает эвристический режим: дубли ищутся по сходству текста, шортсы
выбираются по плотности речи.
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .llm import LLMClient, LLMError
from .transcribe import Transcript, TranscriptSegment

LOGGER = logging.getLogger("avto.edit")

LogFn = Callable[[str], None]

SYSTEM_PROMPT = (
    "Ты — профессиональный видеомонтажёр и редактор. Ты работаешь с "
    "расшифровкой речи на русском языке с таймкодами. Твои ответы должны быть "
    "СТРОГО валидным JSON без пояснений и без markdown-обёрток."
)

CHUNK_SYSTEM = SYSTEM_PROMPT

CHUNK_USER_TEMPLATE = """Ниже — расшифровка речи. Таймкоды указаны ОТНОСИТЕЛЬНО НАЧАЛА ФРАГМЕНТА (00:00 — начало фрагмента).

{transcript}

Твои задачи:
1. Найди места, где спикер оговорился, запнулся, начал фразу и бросил, или повторил одну и ту же мысль несколько раз подряд (дубли).
2. Удали РАННИЕ неудачные дубли и оставь только финальный, самый чистый вариант каждой мысли.
3. Раздели речь на 2-6 смысловых блоков и дай каждому короткий заголовок (для глав ролика).
4. Отметь 1-3 ярких законченных фрагмента, которые хорошо зайдут как короткое видео (длиной 30-55 секунд), с коротким цепляющим заголовком и оценкой от 1 до 10.

Верни строго JSON такого вида:
{{
  "keep_segments": [{{"start": 0.0, "end": 12.5}}],
  "chapters": [{{"time": "00:00", "title": "Вступление"}}],
  "moments": [{{"start": 30.0, "end": 75.0, "title": "Заголовок", "hook": "Цепляющий текст", "score": 8}}]
}}

Требования:
- keep_segments — интервалы (в СЕКУНДАХ, float), которые нужно ОСТАВИТЬ. Охвати ими весь фрагмент, кроме явно удаляемых дублей и оговорок. Не оставляй больших пропусков, если в них нет речи.
- chapters — главы; "time" в формате "MM:SS" (в пределах длительности фрагмента).
- moments — фрагменты длиной 30-55 секунд; "score" — целое число 1..10.
- Если дублей нет — верни keep_segments, покрывающий весь фрагмент."""

SHORTS_SYSTEM = SYSTEM_PROMPT

SHORTS_USER_TEMPLATE = """Ниже — список ярких моментов из ролика (start/end — секунды от начала ролика).

{moments}

Выбери {count} лучших законченных фрагмента для коротких вертикальных роликов (Shorts/Reels).
Длительность каждого — от {min_dur:.0f} до {max_dur:.0f} секунд.
Фрагменты не должны пересекаться и должны идти по хронологии.

Верни строго JSON:
{{"shorts": [{{"start": 0.0, "end": 45.0, "title": "...", "hook": "...", "score": 9}}]}}"""


@dataclass
class Interval:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, float]:
        return {"start": round(self.start, 3), "end": round(self.end, 3)}


@dataclass
class Chapter:
    time: float
    title: str

    def to_dict(self) -> dict[str, Any]:
        return {"time": format_timestamp(self.time), "title": self.title}


@dataclass
class ShortCandidate:
    start: float
    end: float
    title: str = ""
    hook: str = ""
    score: float = 5.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "title": self.title,
            "hook": self.hook,
            "score": self.score,
        }


@dataclass
class EditPlan:
    keep: list[Interval] = field(default_factory=list)
    chapters: list[Chapter] = field(default_factory=list)
    shorts: list[ShortCandidate] = field(default_factory=list)
    source: str = "heuristic"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "keep": [i.to_dict() for i in self.keep],
            "chapters": [c.to_dict() for c in self.chapters],
            "shorts": [s.to_dict() for s in self.shorts],
            "source": self.source,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------
def format_timestamp(value: float) -> str:
    value = max(0.0, float(value))
    minutes = int(value // 60)
    seconds = int(round(value - minutes * 60))
    if seconds == 60:
        minutes += 1
        seconds = 0
    return f"{minutes:02d}:{seconds:02d}"


def parse_timestamp(value: Any) -> float:
    """Принимает 'MM:SS', 'MM:SS.ss', 'H:MM:SS' или число."""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0
    if re.fullmatch(r"[-+]?\d+(\.\d+)?", text):
        return float(text)
    parts = text.split(":")
    try:
        numbers = [float(p) for p in parts]
    except ValueError:
        return 0.0
    if len(numbers) == 3:
        return numbers[0] * 3600 + numbers[1] * 60 + numbers[2]
    if len(numbers) == 2:
        return numbers[0] * 60 + numbers[1]
    return numbers[0]


def normalize_tokens(text: str) -> list[str]:
    cleaned = re.sub(r"[^\w\s]", " ", text.lower(), flags=re.UNICODE)
    return [token for token in cleaned.split() if token]


def similarity(left: list[str], right: list[str]) -> float:
    if not left or not right:
        return 0.0
    matcher = difflib.SequenceMatcher(None, left, right)
    return matcher.ratio()


def merge_intervals(
    intervals: Iterable[Interval], gap: float = 0.35
) -> list[Interval]:
    """Сливает пересекающиеся и почти соприкасающиеся интервалы."""
    ordered = sorted(
        (i for i in intervals if i.duration > 0.01), key=lambda i: i.start
    )
    merged: list[Interval] = []
    for item in ordered:
        if merged and item.start <= merged[-1].end + gap:
            merged[-1].end = max(merged[-1].end, item.end)
        else:
            merged.append(Interval(item.start, item.end))
    return merged


def complement_intervals(keep: list[Interval], duration: float) -> list[Interval]:
    """Отрезки, которые НЕ попали в keep (для оценки полноты покрытия)."""
    gaps: list[Interval] = []
    cursor = 0.0
    for item in sorted(keep, key=lambda i: i.start):
        if item.start > cursor + 0.05:
            gaps.append(Interval(cursor, item.start))
        cursor = max(cursor, item.end)
    if duration > cursor + 0.05:
        gaps.append(Interval(cursor, duration))
    return gaps


# ---------------------------------------------------------------------------
# Разбиение транскрипта на фрагменты
# ---------------------------------------------------------------------------
@dataclass
class TranscriptChunk:
    offset: float
    duration: float
    text: str


def chunk_transcript(
    transcript: Transcript,
    max_chars: int = 18000,
    max_seconds: float = 600.0,
) -> list[TranscriptChunk]:
    """Делит транскрипт на фрагменты, удобные для обработки LLM."""
    chunks: list[TranscriptChunk] = []
    current: list[TranscriptSegment] = []
    current_chars = 0
    current_start = None
    current_end = 0.0

    for seg in transcript.segments:
        if not seg.text.strip():
            continue
        if current_start is None:
            current_start = seg.start
        seg_chars = len(seg.text) + 20
        too_long = current and (
            current_chars + seg_chars > max_chars
            or (seg.end - current_start) > max_seconds
        )
        if too_long:
            chunks.append(
                TranscriptChunk(
                    offset=current_start or 0.0,
                    duration=max(0.0, current_end - (current_start or 0.0)),
                    text=_render_chunk(current, current_start or 0.0),
                )
            )
            current = []
            current_chars = 0
            current_start = seg.start
        current.append(seg)
        current_chars += seg_chars
        current_end = seg.end

    if current:
        chunks.append(
            TranscriptChunk(
                offset=current_start or 0.0,
                duration=max(0.0, current_end - (current_start or 0.0)),
                text=_render_chunk(current, current_start or 0.0),
            )
        )
    return chunks


def _render_chunk(segments: list[TranscriptSegment], offset: float) -> str:
    lines = []
    for seg in segments:
        local_start = max(0.0, seg.start - offset)
        local_end = max(local_start, seg.end - offset)
        lines.append(
            f"[{format_timestamp(local_start)} - {format_timestamp(local_end)}] "
            f"{seg.text.strip()}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Основной планировщик
# ---------------------------------------------------------------------------
def plan_edit(
    transcript: Transcript,
    *,
    llm: LLMClient | None,
    log: LogFn | None = None,
    short_count: int = 3,
    short_min_duration: float = 30.0,
    short_max_duration: float = 55.0,
) -> EditPlan:
    """Строит план монтажа: что оставить, главы и шортсы."""
    emit = log or (lambda message: LOGGER.info(message))
    duration = transcript.duration or (transcript.segments[-1].end if transcript.segments else 0.0)
    plan = EditPlan()

    if llm is None:
        emit("LLM недоступен — работаю в эвристическом режиме.")
        plan.keep = heuristic_keep_segments(transcript, duration)
        plan.chapters = heuristic_chapters(transcript)
        plan.shorts = heuristic_shorts(
            transcript, duration, short_count, short_min_duration, short_max_duration
        )
        plan.source = "heuristic"
        plan.notes.append("Дубли удалены эвристикой (LLM был недоступен)")
        return plan

    chunks = chunk_transcript(transcript)
    if not chunks:
        plan.keep = [Interval(0.0, duration)]
        return plan

    keep: list[Interval] = []
    chapters: list[Chapter] = []
    moments: list[ShortCandidate] = []

    for index, chunk in enumerate(chunks, start=1):
        emit(f"Анализ фрагмента {index}/{len(chunks)} (LLM)...")
        try:
            data = llm.chat_json(CHUNK_SYSTEM, CHUNK_USER_TEMPLATE.format(transcript=chunk.text))
        except LLMError as exc:
            emit(f"Ошибка LLM на фрагменте {index}: {exc}")
            plan.notes.append(f"LLM недоступен на фрагменте {index}: {exc}")
            break

        chunk_keep = _parse_intervals(data.get("keep_segments"), chunk.offset, chunk.duration)
        if not _coverage_ok(chunk_keep, chunk.duration):
            removed = _parse_intervals(data.get("remove_segments"), chunk.offset, chunk.duration)
            if removed:
                chunk_keep = _complement(removed, chunk.offset, chunk.duration)
                emit("  keep_segments неполные — интервалы вычислены как дополнение к remove_segments.")
            else:
                chunk_keep = [Interval(chunk.offset, chunk.offset + chunk.duration)]
                emit("  keep_segments неполные — фрагмент оставлен целиком.")
        keep.extend(chunk_keep)

        for item in data.get("chapters") or []:
            if not isinstance(item, dict):
                continue
            time_value = parse_timestamp(item.get("time", 0))
            title = str(item.get("title", "")).strip()
            if title:
                chapters.append(Chapter(time=chunk.offset + time_value, title=title))

        for item in data.get("moments") or []:
            if not isinstance(item, dict):
                continue
            start = parse_timestamp(item.get("start", 0)) + chunk.offset
            end = parse_timestamp(item.get("end", 0)) + chunk.offset
            if end <= start:
                continue
            moments.append(
                ShortCandidate(
                    start=start,
                    end=end,
                    title=str(item.get("title", "")).strip(),
                    hook=str(item.get("hook", "")).strip(),
                    score=float(item.get("score", 5) or 5),
                )
            )

    if not keep:
        plan.keep = heuristic_keep_segments(transcript, duration)
        plan.source = "heuristic"
        plan.notes.append("LLM не вернул план — применена эвристика")
    else:
        plan.keep = merge_intervals(keep)
        plan.source = "llm"

    plan.chapters = _merge_chapters(chapters, plan.keep, duration)
    if not plan.chapters:
        plan.chapters = heuristic_chapters(transcript)

    plan.shorts = _select_shorts(
        moments,
        transcript,
        duration,
        llm=llm,
        log=emit,
        count=short_count,
        min_duration=short_min_duration,
        max_duration=short_max_duration,
    )
    return plan


def _parse_intervals(
    raw: Any, offset: float, chunk_duration: float
) -> list[Interval]:
    if not isinstance(raw, list):
        return []
    items: list[Interval] = []
    for entry in raw:
        if isinstance(entry, dict):
            start = parse_timestamp(entry.get("start", 0))
            end = parse_timestamp(entry.get("end", 0))
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            start = parse_timestamp(entry[0])
            end = parse_timestamp(entry[1])
        else:
            continue
        if end <= start:
            continue
        # подрезаем по границам фрагмента — модель могла «вылезти» за них
        start = max(0.0, min(start, chunk_duration))
        end = max(0.0, min(end, chunk_duration))
        if end - start < 0.05:
            continue
        items.append(Interval(offset + start, offset + end))
    return merge_intervals(items)


def _complement(removed: list[Interval], offset: float, chunk_duration: float) -> list[Interval]:
    result: list[Interval] = []
    cursor = offset
    for item in sorted(removed, key=lambda i: i.start):
        if item.start > cursor:
            result.append(Interval(cursor, item.start))
        cursor = max(cursor, item.end)
    end = offset + chunk_duration
    if end > cursor:
        result.append(Interval(cursor, end))
    return result


def _coverage_ok(keep: list[Interval], chunk_duration: float) -> bool:
    """Слишком маленькое покрытие — признак того, что модель ответила не тем.

    Порог снижен до 30%: модель вправе вырезать большую часть ролика
    (например, если половина — это повторы), но совсем пустой результат
    считаем ошибкой и перестраховываемся эвристикой.
    """
    if not keep or chunk_duration <= 0:
        return False
    covered = sum(item.duration for item in keep)
    return covered >= chunk_duration * 0.3


def _merge_chapters(
    chapters: list[Chapter], keep: list[Interval], duration: float
) -> list[Chapter]:
    if not chapters:
        return []
    chapters.sort(key=lambda c: c.time)
    merged: list[Chapter] = []
    for chapter in chapters:
        chapter.time = max(0.0, min(chapter.time, duration))
        if merged and chapter.time - merged[-1].time < 1.0:
            continue
        merged.append(chapter)
    # главы должны попадать внутрь оставленных интервалов
    if keep:
        merged = [c for c in merged if any(k.start <= c.time <= k.end for k in keep)]
    return merged


# ---------------------------------------------------------------------------
# Выбор шортсов
# ---------------------------------------------------------------------------
def _select_shorts(
    moments: list[ShortCandidate],
    transcript: Transcript,
    duration: float,
    *,
    llm: LLMClient | None,
    log: LogFn,
    count: int,
    min_duration: float,
    max_duration: float,
) -> list[ShortCandidate]:
    if not moments:
        return heuristic_shorts(transcript, duration, count, min_duration, max_duration)

    # 1) пробуем финальный отбор на LLM
    if llm is not None:
        rendered = "\n".join(
            f"- start={m.start:.1f} end={m.end:.1f} "
            f"({m.duration:.0f}s) «{m.title}» {m.hook}"
            for m in moments
        )
        try:
            data = llm.chat_json(
                SHORTS_SYSTEM,
                SHORTS_USER_TEMPLATE.format(
                    moments=rendered,
                    count=count,
                    min_dur=min_duration,
                    max_dur=max_duration,
                ),
            )
            picked: list[ShortCandidate] = []
            for item in data.get("shorts") or []:
                if not isinstance(item, dict):
                    continue
                start = parse_timestamp(item.get("start", 0))
                end = parse_timestamp(item.get("end", 0))
                if end - start < min_duration * 0.6:
                    continue
                picked.append(
                    ShortCandidate(
                        start=start,
                        end=end,
                        title=str(item.get("title", "")).strip(),
                        hook=str(item.get("hook", "")).strip(),
                        score=float(item.get("score", 5) or 5),
                    )
                )
            picked = _enforce_short_rules(picked, duration, count, min_duration, max_duration)
            if picked:
                log(f"LLM выбрал {len(picked)} шортсов.")
                return picked
        except LLMError as exc:
            log(f"Не удалось получить шортсы от LLM: {exc}")

    # 2) берём моменты с наивысшей оценкой
    ordered = sorted(moments, key=lambda m: m.score, reverse=True)
    picked = _enforce_short_rules(ordered, duration, count, min_duration, max_duration)
    if picked:
        log(f"Шортсы выбраны по оценкам моментов: {len(picked)}.")
        return picked

    # 3) эвристика
    return heuristic_shorts(transcript, duration, count, min_duration, max_duration)


def _enforce_short_rules(
    candidates: list[ShortCandidate],
    duration: float,
    count: int,
    min_duration: float,
    max_duration: float,
) -> list[ShortCandidate]:
    """Оставляет непересекающиеся шортсы нужной длины, по хронологии."""
    result: list[ShortCandidate] = []
    for cand in sorted(candidates, key=lambda c: c.start):
        if len(result) >= count:
            break
        start = max(0.0, cand.start)
        end = min(duration, cand.end) if duration else cand.end
        if end - start > max_duration:
            end = start + max_duration
        if end - start < min_duration:
            continue
        if any(
            start < existing.end + 2.0 and end > existing.start - 2.0
            for existing in result
        ):
            continue
        result.append(
            ShortCandidate(
                start=start,
                end=end,
                title=cand.title,
                hook=cand.hook,
                score=cand.score,
            )
        )
    return result


# ---------------------------------------------------------------------------
# Эвристики (без LLM)
# ---------------------------------------------------------------------------
def heuristic_keep_segments(
    transcript: Transcript, duration: float
) -> list[Interval]:
    """Убирает повторы и оговорки по сходству соседних фраз.

    Работает на уровне отдельных предложений: режется ранняя, «запнутая»
    версия фразы, а финальная (обычно более полная) остаётся.
    """
    units = _split_units(transcript.segments)
    if not units:
        return [Interval(0.0, duration)] if duration else []

    removed: list[Interval] = []
    for index in range(1, len(units)):
        previous = units[index - 1]
        current = units[index]
        left = normalize_tokens(previous.text)
        right = normalize_tokens(current.text)
        if len(left) < 3 or len(right) < 3:
            continue

        ratio = similarity(left, right)
        prefix_match = _common_prefix_len(left, right)
        tail_overlap = _tail_overlap(left, right)

        is_duplicate = (
            ratio >= 0.72
            or prefix_match >= max(3, int(len(left) * 0.6))
            or tail_overlap >= max(3, int(min(len(left), len(right)) * 0.6))
        )
        if is_duplicate:
            # оставляем финальный (более чистый) вариант
            removed.append(Interval(previous.start, previous.end))

    if not removed:
        return [Interval(0.0, duration)] if duration else []

    # защита: не вырезаем больше 40% ролика
    removed_total = sum(item.duration for item in removed)
    if duration and removed_total > duration * 0.4:
        return [Interval(0.0, duration)]

    keep = _complement(removed, 0.0, duration)
    return merge_intervals(keep, gap=0.2)


def _split_units(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    """Делит сегменты на предложения с пропорциональным распределением времени."""
    units: list[TranscriptSegment] = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        parts = [part.strip() for part in re.split(r"(?<=[.!?…])\s+", text) if part.strip()]
        if not parts:
            parts = [text]
        if len(parts) == 1:
            units.append(TranscriptSegment(start=seg.start, end=seg.end, text=parts[0]))
            continue
        span = max(0.0, seg.end - seg.start)
        total = sum(len(part) for part in parts) or 1
        cursor = seg.start
        for part in parts:
            length = span * (len(part) / total)
            units.append(
                TranscriptSegment(start=cursor, end=cursor + length, text=part)
            )
            cursor += length
    return units


def _common_prefix_len(left: list[str], right: list[str]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def _tail_overlap(left: list[str], right: list[str]) -> int:
    """Сколько слов в конце left повторяется в начале right."""
    best = 0
    for size in range(min(len(left), len(right)), 2, -1):
        if left[-size:] == right[:size]:
            best = size
            break
    return best


def heuristic_chapters(transcript: Transcript) -> list[Chapter]:
    """Делит ролик на равномерные смысловые блоки по объёму текста."""
    segments = [s for s in transcript.segments if s.text.strip()]
    if not segments:
        return []
    total_words = sum(len(normalize_tokens(s.text)) for s in segments) or 1
    target = max(1, total_words // 4)
    chapters: list[Chapter] = []
    accumulated = 0
    start = segments[0].start
    title = ""
    for seg in segments:
        if not title:
            title = _short_title(seg.text)
        accumulated += len(normalize_tokens(seg.text))
        if accumulated >= target and len(chapters) < 6:
            chapters.append(Chapter(time=start, title=title or "Блок"))
            start = seg.end
            title = ""
            accumulated = 0
    if title:
        chapters.append(Chapter(time=start, title=title))
    return chapters[:7]


def _short_title(text: str, words: int = 6) -> str:
    tokens = text.split()
    snippet = " ".join(tokens[:words]).strip(" ,.—-")
    if len(tokens) > words:
        snippet += "…"
    return snippet or "Блок"


def heuristic_shorts(
    transcript: Transcript,
    duration: float,
    count: int,
    min_duration: float,
    max_duration: float,
) -> list[ShortCandidate]:
    """Выбирает шортсы по плотности речи (окна с максимумом слов в секунду)."""
    segments = [s for s in transcript.segments if s.text.strip()]
    if not segments or duration <= 0:
        return []

    words_per_segment = [max(1, len(normalize_tokens(s.text))) for s in segments]
    max_duration = max(min_duration, min(max_duration, duration))
    window = min_duration
    best: list[tuple[float, float, float]] = []  # (score, start, end)

    for start in [i * 1.0 for i in range(0, int(duration) + 1)]:
        end = start + window
        if end > duration:
            break
        score = 0.0
        for seg, weight in zip(segments, words_per_segment):
            overlap = min(seg.end, end) - max(seg.start, start)
            if overlap > 0:
                score += weight * (overlap / max(0.001, seg.end - seg.start))
        best.append((score, start, end))

    if not best:
        return []

    best.sort(key=lambda item: item[0], reverse=True)
    chosen: list[ShortCandidate] = []
    for score, start, end in best:
        if len(chosen) >= count:
            break
        if any(start < c.end + 5 and end > c.start - 5 for c in chosen):
            continue
        title = _short_title(_text_in_range(segments, start, end))
        chosen.append(
            ShortCandidate(
                start=start,
                end=end,
                title=title,
                hook=title,
                score=round(score / max(1.0, window), 2),
            )
        )
    return sorted(chosen, key=lambda c: c.start)


def _text_in_range(
    segments: list[TranscriptSegment], start: float, end: float
) -> str:
    parts = [
        seg.text
        for seg in segments
        if seg.end > start and seg.start < end
    ]
    return " ".join(parts)
