"""Download a YouTube video, transcribe with WhisperX, write a cleaned `.txt`.

WhisperX needs torch 1.13.1 (cu117), which conflicts with the classifier's
PyTorch ≥ 2.0. The README documents the recommended `whisper-env`
virtualenv; this module assumes the `whisperx` binary is reachable either
on PATH or via the `WHISPERX_BIN` environment variable.

CLI:
    python -m src.extract_subtitles "https://youtu.be/<id>" --out-dir output/
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import yt_dlp


def download_audio(url: str, out_dir: Path, basename: str = "audio") -> Path:
    """Pull the best audio track from `url` and convert it to MP3.

    Returns the MP3 path (`out_dir / f"{basename}.mp3"`).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3"},
        ],
        "outtmpl": str(out_dir / basename),
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return out_dir / f"{basename}.mp3"


def _resolve_whisperx_bin(explicit: Optional[str]) -> str:
    candidate = (
        explicit
        or os.environ.get("WHISPERX_BIN")
        or shutil.which("whisperx")
    )
    if candidate:
        return candidate
    raise FileNotFoundError(
        "whisperx binary not found.\n"
        "Set it up once in an isolated virtualenv (it requires torch 1.13.1):\n"
        "  virtualenv whisper-env\n"
        "  whisper-env/bin/pip install git+https://github.com/m-bain/whisperx.git@v2.0.1\n"
        "Then run with WHISPERX_BIN=whisper-env/bin/whisperx, or pass --whisperx-bin."
    )


def transcribe(
    mp3_path: Path,
    *,
    language: str = "en",
    model: str = "large-v2",
    output_dir: Path = Path("."),
    whisperx_bin: Optional[str] = None,
) -> Path:
    """Run WhisperX on `mp3_path` and return the produced `.srt` path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        _resolve_whisperx_bin(whisperx_bin),
        str(mp3_path),
        "--language", language,
        "--model", model,
        "--vad_filter", "True",
        "--output_format", "srt",
        "--output_dir", str(output_dir),
        "--align_model", "WAV2VEC2_ASR_LARGE_LV60K_960H",
    ]
    subprocess.run(cmd, check=True)
    return output_dir / f"{mp3_path.stem}.srt"


def srt_to_text(srt_path: Path, txt_path: Path) -> Path:
    """Strip SRT cue numbers and timestamps to plain text, one line per cue."""
    lines = []
    for raw in srt_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.isdigit() or "-->" in line:
            continue
        lines.append(line)
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return txt_path


def extract(url: str, out_dir: Path, **kwargs) -> Path:
    """End-to-end: YouTube URL → cleaned `.txt`. Returns the `.txt` path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mp3 = download_audio(url, out_dir)
    srt = transcribe(mp3, output_dir=out_dir, **kwargs)
    return srt_to_text(srt, out_dir / f"{srt.stem}.txt")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YouTube → caption → cleaned text via yt-dlp + WhisperX."
    )
    p.add_argument("url", help="YouTube video URL")
    p.add_argument("--out-dir", type=Path, default=Path("./output"))
    p.add_argument("--language", default="en")
    p.add_argument("--model", default="large-v2")
    p.add_argument("--whisperx-bin", default=None)
    return p.parse_args()


def main() -> None:
    a = _parse_args()
    txt = extract(
        a.url,
        a.out_dir,
        language=a.language,
        model=a.model,
        whisperx_bin=a.whisperx_bin,
    )
    print(f"txt: {txt}")


if __name__ == "__main__":
    main()
