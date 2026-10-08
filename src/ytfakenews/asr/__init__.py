"""Speech-to-text: turn a video URL or a local media file into a timed transcript.

- :mod:`ytfakenews.asr.youtube` downloads the audio or fetches the captions of a video
  with yt-dlp;
- :mod:`ytfakenews.asr.whisper` transcribes audio with faster-whisper;
- :mod:`ytfakenews.asr.transcript` defines :class:`Transcript` and saves it as
  ``.txt``, ``.srt`` and ``.json``.

:func:`transcribe_source` combines them, as the ``transcribe`` and ``run`` commands do.
yt-dlp and faster-whisper make up the ``asr`` extra and are imported on first use. Only
``youtube._extract_info`` and ``whisper._load_whisper_model`` reach the network, and
the tests replace exactly these two functions.
"""

from __future__ import annotations

import logging
import re
import tempfile
from dataclasses import replace
from pathlib import Path

from ytfakenews.asr.transcript import Transcript, TranscriptFiles, VideoInfo, save_transcript
from ytfakenews.asr.whisper import DEFAULT_WHISPER_MODEL, transcribe
from ytfakenews.asr.youtube import DownloadedAudio, download_audio, fetch_captions
from ytfakenews.errors import TranscriptionError

__all__ = [
    "DEFAULT_TRANSCRIPT_DIR",
    "DEFAULT_WHISPER_MODEL",
    "DownloadedAudio",
    "Transcript",
    "TranscriptFiles",
    "VideoInfo",
    "download_audio",
    "fetch_captions",
    "save_transcript",
    "transcribe",
    "transcribe_source",
]

logger = logging.getLogger(__name__)

DEFAULT_TRANSCRIPT_DIR = Path("outputs")
"""Where ``transcribe`` and ``run`` save ``<video id>.txt``, ``.srt`` and ``.json``."""


def transcribe_source(
    source: str,
    *,
    output_dir: str | Path = DEFAULT_TRANSCRIPT_DIR,
    captions: bool = False,
    language: str | None = None,
    translate: bool = False,
    model_size: str = DEFAULT_WHISPER_MODEL,
    device: str = "auto",
    compute_type: str = "auto",
    keep_audio: bool = False,
) -> tuple[Transcript, TranscriptFiles]:
    """Transcribe a video URL or a local audio/video file and save the transcript.

    For a URL the audio goes to a temporary directory unless ``keep_audio`` is set, in
    which case it is kept in ``output_dir``. With ``captions=True`` the video's own
    subtitles in ``language`` (default ``en``) are used instead of Whisper.
    """
    output_dir = Path(output_dir)

    def whisper(audio: Path) -> Transcript:
        return transcribe(
            audio,
            model_size=model_size,
            language=language,
            translate=translate,
            device=device,
            compute_type=compute_type,
        )

    local = Path(source)
    if local.is_file():
        if captions:
            raise TranscriptionError("--captions needs a video URL, not a local file")
        transcript = whisper(local)
        stem = local.stem
    elif re.match(r"^https?://", source):
        if captions:
            transcript = fetch_captions(source, language=language or "en")
        else:
            with tempfile.TemporaryDirectory(prefix="ytfakenews-") as tmp:
                audio = download_audio(source, output_dir if keep_audio else tmp)
                transcript = replace(whisper(audio.path), video=audio.video)
        stem = transcript.video.id if transcript.video else "transcript"
    else:
        raise TranscriptionError(f"{source!r} is neither an existing file nor an http(s) URL")
    files = save_transcript(transcript, output_dir, stem)
    logger.info("Wrote %s (%d segments)", files.txt, len(transcript.segments))
    return transcript, files
