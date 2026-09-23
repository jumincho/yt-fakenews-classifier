from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest

from tests.helpers import FIXTURES
from ytfakenews.text import (
    Segment,
    chunk_spans,
    chunk_words,
    clean_text,
    format_timestamp,
    parse_srt,
    parse_timestamp,
    parse_vtt,
    read_transcript,
    segments_to_srt,
    segments_to_text,
)

# ----------------------------------------------------------------------------- timestamps


@pytest.mark.parametrize(
    ("value", "seconds"),
    [
        ("00:00:01,000", 1.0),
        ("01:02:03,450", 3723.45),
        ("02:03.450", 123.45),
        ("1:00:00.5", 3600.5),
        ("00:10", 10.0),
    ],
)
def test_parse_timestamp(value: str, seconds: float) -> None:
    assert parse_timestamp(value) == pytest.approx(seconds)


@pytest.mark.parametrize("value", ["", "12", "a:b:c", "1:2:3:4"])
def test_parse_timestamp_rejects_garbage(value: str) -> None:
    with pytest.raises(ValueError, match="invalid timestamp"):
        parse_timestamp(value)


def test_format_timestamp() -> None:
    assert format_timestamp(3723.45) == "01:02:03,450"
    assert format_timestamp(0.0005) == "00:00:00,000"
    assert format_timestamp(59.9996, decimal_marker=".") == "00:01:00.000"
    with pytest.raises(ValueError, match="non-negative"):
        format_timestamp(-1)


# ---------------------------------------------------------------------------------- SRT


def test_parse_srt_strips_markup_and_joins_lines() -> None:
    segments = parse_srt((FIXTURES / "sample.srt").read_text(encoding="utf-8"))
    assert segments == [
        Segment(1.0, 3.5, "Good evening, and welcome."),
        Segment(3.6, 6.0, "Tonight: the city council votes on the budget."),
        Segment(6.1, 7.0, "[Music]"),
    ]


def test_parse_srt_handles_crlf_bom_and_missing_blank_lines() -> None:
    content = (
        "\ufeff1\r\n00:00:00,000 --> 00:00:01,000\r\nfirst line\r\n"
        "2\r\n00:00:01,000 --> 00:00:02,000\r\nsecond line\r\n"
        "   \r\n3\r\n00:00:02,000 --> 00:00:03,000\r\n\r\n"
    )
    assert parse_srt(content) == [
        Segment(0.0, 1.0, "first line"),
        Segment(1.0, 2.0, "second line"),
    ]


def test_srt_round_trip() -> None:
    segments = [Segment(0.0, 1.25, "hello there"), Segment(61.5, 3723.45, "and goodbye")]
    rendered = segments_to_srt(segments)
    assert rendered.startswith("1\n00:00:00,000 --> 00:00:01,250\nhello there\n\n2\n")
    assert parse_srt(rendered) == segments


# ------------------------------------------------------------------------------- WebVTT


def test_parse_vtt_dedupes_youtube_rolling_captions() -> None:
    segments = parse_vtt((FIXTURES / "youtube_auto.vtt").read_text(encoding="utf-8"))
    assert [segment.text for segment in segments] == [
        "good evening and welcome",
        "to the evening news",
        ">> the council met today",
        "[Music]",
    ]
    assert segments[0].start == 0.0
    assert segments[1].start == pytest.approx(2.32)


def test_parse_vtt_without_dedupe_keeps_every_cue() -> None:
    content = (FIXTURES / "youtube_auto.vtt").read_text(encoding="utf-8")
    texts = [segment.text for segment in parse_vtt(content, dedupe=False)]
    assert texts[1] == "good evening and welcome"  # the transitional repeat
    assert texts[2] == "good evening and welcome to the evening news"
    assert len(texts) == 7


def test_parse_vtt_dedupes_lines_that_grow_word_by_word() -> None:
    content = (
        "WEBVTT\n\n"
        "00:00.000 --> 00:01.000\nthe mayor\n\n"
        "00:01.000 --> 00:02.000\nthe mayor said\n\n"
        "00:02.000 --> 00:03.000\nthe mayor said no\n\n"
        "00:03.000 --> 00:04.000\nthe mayor said no\nto the plan\n"
    )
    assert segments_to_text(parse_vtt(content)) == "the mayor\nsaid\nno\nto the plan"


def test_parse_vtt_skips_header_note_style_and_identifiers() -> None:
    content = (
        "WEBVTT - some title\n\n"
        "STYLE\n::cue { color: yellow }\n\n"
        "NOTE this is a comment\nspanning two lines\n\n"
        "intro\n00:00:00.000 --> 00:00:02.000 line:0\n<v Anna>Hello &amp; welcome</v>\n\n"
        "00:00:02.000 --> 00:00:03.000\nsecond cue\n"
    )
    assert [segment.text for segment in parse_vtt(content)] == ["Hello & welcome", "second cue"]


def test_read_transcript_dispatches_on_suffix(tmp_path: Path) -> None:
    assert read_transcript(FIXTURES / "sample.srt").splitlines()[0] == "Good evening, and welcome."
    assert read_transcript(FIXTURES / "youtube_auto.vtt").startswith("good evening and welcome\n")
    plain = tmp_path / "notes.txt"
    plain.write_text("just text\n", encoding="utf-8")
    assert read_transcript(plain) == "just text\n"


# ------------------------------------------------------------------------------ cleanup


def test_clean_text_removes_caption_artifacts() -> None:
    raw = (
        "[Music]  Good evening &amp; welcome.\n>> Tonight <i>only</i>: ♪ la la ♪\n"
        "00:00:01,000 --> 00:00:02,000\n[APPLAUSE] Back at 10:30 on 00:01:02.500 channel 4."
    )
    assert clean_text(raw) == (
        "Good evening & welcome. Tonight only: la la Back at 10:30 on channel 4."
    )


def test_clean_text_keeps_comparisons_and_empty_input() -> None:
    assert clean_text("if a < b and b > c") == "if a < b and b > c"
    assert clean_text("  \n\t ") == ""


# ----------------------------------------------------------------------------- chunking


def test_short_texts_are_a_single_chunk() -> None:
    assert chunk_spans(0) == []
    assert chunk_spans(1) == [(0, 1)]
    assert chunk_spans(300, max_words=300, overlap=50) == [(0, 300)]


@pytest.mark.parametrize("n_words", [301, 551, 760, 1000, 5003])
@pytest.mark.parametrize(("max_words", "overlap"), [(300, 50), (100, 0), (50, 49)])
def test_chunk_spans_cover_everything_with_full_windows(
    n_words: int, max_words: int, overlap: int
) -> None:
    spans = chunk_spans(n_words, max_words=max_words, overlap=overlap)
    assert spans[0][0] == 0
    assert spans[-1][1] == n_words
    assert all(end - start == max_words for start, end in spans)
    for (_, previous_end), (next_start, _) in pairwise(spans):
        assert previous_end - next_start >= overlap
    # Minimal: one window fewer could not keep the required overlap.
    stride = max_words - overlap
    assert (len(spans) - 2) * stride + max_words < n_words


def test_chunk_spans_example() -> None:
    assert chunk_spans(760, max_words=300, overlap=50) == [(0, 300), (230, 530), (460, 760)]


@pytest.mark.parametrize(("max_words", "overlap"), [(0, 0), (10, 10), (10, -1)])
def test_chunk_spans_validates_arguments(max_words: int, overlap: int) -> None:
    with pytest.raises(ValueError, match="must"):
        chunk_spans(100, max_words=max_words, overlap=overlap)


def test_chunk_words_splits_text() -> None:
    text = " ".join(f"w{i}" for i in range(10))
    assert chunk_words(text, max_words=4, overlap=1) == [
        "w0 w1 w2 w3",
        "w3 w4 w5 w6",
        "w6 w7 w8 w9",
    ]
    assert chunk_words("   ") == []
