from __future__ import annotations

import functools
import json
import sys
import threading
from collections.abc import Iterator
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, ClassVar

import pytest

from tests.helpers import VIDEO, VIDEO_URL, FakeYouTube, import_or_skip, write_tone
from ytfakenews import transcribe as tr
from ytfakenews.errors import MissingDependencyError, TranscriptionError
from ytfakenews.text import Segment, parse_srt

# ------------------------------------------------------------------- with fake layers


def test_download_audio_requests_audio_only(fake_youtube: FakeYouTube, tmp_path: Path) -> None:
    result = tr.download_audio(VIDEO_URL, tmp_path)
    assert result.path == tmp_path / f"{VIDEO['id']}.webm"
    assert result.path.is_file()
    assert result.video.title == "Evening news"
    assert result.video.url == VIDEO_URL

    url, options, download = fake_youtube.extract_calls[0]
    assert (url, download) == (VIDEO_URL, True)
    assert options["format"] == "bestaudio/best"
    assert "postprocessors" not in options  # no ffmpeg needed


def test_transcribe_collects_segments(fake_youtube: FakeYouTube, tmp_path: Path) -> None:
    audio = write_tone(tmp_path / "clip.wav")
    transcript = tr.transcribe(audio, model_size="tiny", device="cpu")

    assert transcript.segments == (
        Segment(0.0, 4.0, "Good evening, officials said today."),
        Segment(8.0, 12.0, "The committee report is out."),
    )
    assert transcript.source == "whisper"
    assert transcript.language == "en"
    assert transcript.details["language_probability"] == 0.987
    assert fake_youtube.whisper_loads == [
        {"model_size": "tiny", "device": "cpu", "compute_type": "auto"}
    ]
    _, kwargs = fake_youtube.whisper_calls[0]
    assert kwargs == {"language": None, "task": "transcribe", "beam_size": 5, "vad_filter": True}


def test_transcribe_can_translate(fake_youtube: FakeYouTube, tmp_path: Path) -> None:
    fake_youtube.spoken_language = "ko"
    transcript = tr.transcribe(write_tone(tmp_path / "clip.wav"), language="ko", translate=True)
    assert transcript.language == "en"
    assert transcript.details["spoken_language"] == "ko"
    assert fake_youtube.whisper_calls[0][1]["task"] == "translate"


def test_transcribe_source_for_a_url(fake_youtube: FakeYouTube, tmp_path: Path) -> None:
    transcript, files = tr.transcribe_source(VIDEO_URL, output_dir=tmp_path)
    assert transcript.video is not None
    assert transcript.video.id == VIDEO["id"]
    assert files.txt == tmp_path / f"{VIDEO['id']}.txt"
    # The audio went to a temporary directory and is gone.
    assert sorted(path.suffix for path in tmp_path.iterdir()) == [".json", ".srt", ".txt"]

    tr.transcribe_source(VIDEO_URL, output_dir=tmp_path, keep_audio=True)
    assert (tmp_path / f"{VIDEO['id']}.webm").is_file()


def test_transcribe_source_for_a_local_file(fake_youtube: FakeYouTube, tmp_path: Path) -> None:
    audio = write_tone(tmp_path / "interview.wav")
    transcript, files = tr.transcribe_source(str(audio), output_dir=tmp_path / "out")
    assert transcript.video is None
    assert files.srt.name == "interview.srt"
    assert fake_youtube.extract_calls == []


def test_transcribe_source_rejects_bad_sources(fake_youtube: FakeYouTube, tmp_path: Path) -> None:
    with pytest.raises(TranscriptionError, match="neither an existing file nor"):
        tr.transcribe_source("not-a-url", output_dir=tmp_path)
    audio = write_tone(tmp_path / "clip.wav")
    with pytest.raises(TranscriptionError, match="--captions needs a video URL"):
        tr.transcribe_source(str(audio), output_dir=tmp_path, captions=True)


def test_fetch_captions_uses_automatic_captions(fake_youtube: FakeYouTube) -> None:
    transcript = tr.fetch_captions(VIDEO_URL)
    assert transcript.source == "captions"
    assert transcript.details == {"track": "en", "kind": "automatic", "machine_translated": False}
    assert transcript.text.splitlines() == [
        "good evening and welcome",
        "to the evening news",
        ">> the council met today",
        "[Music]",
    ]
    options = fake_youtube.extract_calls[0][1]
    assert options["skip_download"] is True
    assert options["subtitleslangs"] == ["en", "en-.+"]


def test_fetch_captions_prefers_manual_subtitles(fake_youtube: FakeYouTube) -> None:
    fake_youtube.manual_tracks = {"en-GB": "WEBVTT\n\n00:00.000 --> 00:02.000\nHand-made line\n"}
    transcript = tr.fetch_captions(VIDEO_URL)
    assert transcript.details["track"] == "en-GB"
    assert transcript.details["kind"] == "manual"
    assert transcript.language == "en-GB"
    assert transcript.text == "Hand-made line"


def test_fetch_captions_flags_machine_translation(fake_youtube: FakeYouTube) -> None:
    vtt = "WEBVTT\n\n00:00.000 --> 00:02.000\nTranslated line\n"
    fake_youtube.automatic_tracks = {"ko-orig": vtt, "ko": vtt, "en": vtt}
    transcript = tr.fetch_captions(VIDEO_URL)
    assert transcript.details == {"track": "en", "kind": "automatic", "machine_translated": True}


def test_fetch_captions_without_matching_track(fake_youtube: FakeYouTube) -> None:
    with pytest.raises(TranscriptionError, match="no 'fr' captions"):
        tr.fetch_captions(VIDEO_URL, language="fr")


def test_save_transcript_writes_three_files(tmp_path: Path) -> None:
    transcript = tr.Transcript(
        segments=(Segment(0.0, 1.5, "Hello"), Segment(1.5, 3.0, "world")),
        source="whisper",
        language="en",
        details={"model": "small"},
        video=tr.VideoInfo(id="vid"),
    )
    files = tr.save_transcript(transcript, tmp_path, "a/b c")
    assert files.txt.name == "a_b_c.txt"
    assert files.txt.read_text(encoding="utf-8") == "Hello\nworld\n"
    assert parse_srt(files.srt.read_text(encoding="utf-8")) == list(transcript.segments)
    payload = json.loads(files.json.read_text(encoding="utf-8"))
    assert payload["video"]["id"] == "vid"
    assert payload["segments"][1] == {"start": 1.5, "end": 3.0, "text": "world"}


# ------------------------------------------------------- boundaries with stub modules


def _stub_yt_dlp(info: dict[str, Any] | None = None, error: str | None = None) -> ModuleType:
    module = ModuleType("yt_dlp")

    class DownloadError(Exception):
        pass

    class YoutubeDL:
        last_params: ClassVar[dict[str, Any]] = {}

        def __init__(self, params: dict[str, Any]) -> None:
            YoutubeDL.last_params = params

        def __enter__(self) -> YoutubeDL:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def extract_info(self, url: str, download: bool) -> dict[str, Any]:
            if error:
                raise DownloadError(error)
            return dict(info or {}, url=url, download=download)

        @staticmethod
        def sanitize_info(value: dict[str, Any]) -> dict[str, Any]:
            return value

    module.YoutubeDL = YoutubeDL
    module.utils = SimpleNamespace(DownloadError=DownloadError)
    return module


def test_extract_info_calls_yt_dlp(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _stub_yt_dlp(info={"id": "x"})
    monkeypatch.setitem(sys.modules, "yt_dlp", stub)
    info = tr._extract_info("https://example.com/v", {"format": "bestaudio"}, download=True)
    assert info == {"id": "x", "url": "https://example.com/v", "download": True}
    params = stub.YoutubeDL.last_params
    assert params["format"] == "bestaudio"
    assert params["noplaylist"] is True
    assert params["quiet"] is True


def test_extract_info_converts_download_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "yt_dlp", _stub_yt_dlp(error="ERROR: Video unavailable"))
    with pytest.raises(TranscriptionError, match=r"yt-dlp failed for u: Video unavailable$"):
        tr._extract_info("u", {}, download=False)


def test_load_whisper_model(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[tuple[str, dict[str, Any]]] = []

    def whisper_model(size: str, **kwargs: Any) -> str:
        if size == "broken":
            raise RuntimeError("CUDA driver not found")
        created.append((size, kwargs))
        return "model"

    stub = ModuleType("faster_whisper")
    stub.WhisperModel = whisper_model
    monkeypatch.setitem(sys.modules, "faster_whisper", stub)
    assert tr._load_whisper_model("small", device="auto", compute_type="int8") == "model"
    assert created == [("small", {"device": "auto", "compute_type": "int8"})]
    with pytest.raises(TranscriptionError, match="CUDA driver not found"):
        tr._load_whisper_model("broken", device="cuda", compute_type="auto")


def test_missing_asr_extra_is_explained(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "yt_dlp", None)
    with pytest.raises(MissingDependencyError, match=r"'asr' dependencies") as info:
        tr.download_audio(VIDEO_URL, tmp_path)
    assert 'pip install -e ".[asr]"' in str(info.value)
    assert isinstance(info.value, ImportError)


# --------------------------------------------------- real libraries, still offline


@pytest.fixture
def local_media_server(tmp_path: Path) -> Iterator[str]:
    """Serve ``tmp_path/media`` over HTTP on localhost."""
    media = tmp_path / "media"
    media.mkdir()
    write_tone(media / "tone.wav")
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(media))
    handler.log_message = lambda *args: None
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    thread.join()


def test_real_yt_dlp_downloads_audio(local_media_server: str, tmp_path: Path) -> None:
    import_or_skip("yt_dlp")
    result = tr.download_audio(f"{local_media_server}/tone.wav", tmp_path / "out")
    assert result.path.read_bytes() == (tmp_path / "media" / "tone.wav").read_bytes()
    assert result.video.id == "tone"


def test_real_faster_whisper_decodes_audio_without_ffmpeg(tmp_path: Path) -> None:
    faster_whisper = import_or_skip("faster_whisper")
    samples = faster_whisper.decode_audio(str(write_tone(tmp_path / "tone.wav", seconds=0.5)))
    assert samples.shape == (8000,)
