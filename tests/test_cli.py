from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

from tests.helpers import FIXTURES, VIDEO, VIDEO_URL, FakeYouTube
from ytfakenews import __version__
from ytfakenews.artifacts import write_manifest
from ytfakenews.cli import main


def run_cli(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"ytfakenews {__version__}"


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], "COMMAND"),
        (["train"], "baseline"),
        (["train", "baseline"], "--C"),
        (["evaluate"], "--chunked"),
        (["predict"], "--text"),
        (["transcribe"], "--captions"),
        (["run"], "--whisper-model"),
    ],
)
def test_help(capsys: pytest.CaptureFixture[str], argv: list[str], expected: str) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main([*argv, "--help"])
    assert exit_info.value.code == 0
    assert expected in capsys.readouterr().out


def test_missing_command_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main([])
    assert exit_info.value.code == 2
    assert "required" in capsys.readouterr().err


def test_train_baseline(capsys: pytest.CaptureFixture[str], news_zip: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "model"
    code, out, err = run_cli(
        capsys, "train", "baseline", "--data", str(news_zip), "--output-dir", str(out_dir)
    )
    assert code == 0
    assert f"Saved baseline model to {out_dir}" in out
    assert "validation" in out
    assert "Test confusion matrix" in out
    assert "Loaded 60 rows" in err
    assert (out_dir / "model.joblib").is_file()


def test_evaluate(capsys: pytest.CaptureFixture[str], baseline_dir: Path, tmp_path: Path) -> None:
    code, out, _ = run_cli(capsys, "evaluate", "--model", str(baseline_dir))
    assert code == 0
    assert "on the test split (document)" in out

    report = tmp_path / "report.json"
    args = ("--split", "val", "--chunked", "--chunk-words", "20", "--overlap", "5", "--json")
    code, out, _ = run_cli(
        capsys, "evaluate", "--model", str(baseline_dir), *args, "--output", str(report)
    )
    assert code == 0
    metrics = json.loads(out)
    assert metrics == json.loads(report.read_text(encoding="utf-8"))
    assert metrics["split"] == "val"
    assert metrics["input"] == {"mode": "chunked", "chunk_words": 20, "overlap": 5}
    assert metrics["model"] == {"path": baseline_dir.as_posix(), "backend": "baseline"}


def test_evaluate_short_inputs(capsys: pytest.CaptureFixture[str], baseline_dir: Path) -> None:
    code, out, _ = run_cli(capsys, "evaluate", "--model", str(baseline_dir), "--max-words", "5")
    assert code == 0
    assert "on the test split (document, first 5 words)" in out
    code, out, _ = run_cli(
        capsys, "evaluate", "--model", str(baseline_dir), "--max-words", "5", "--json"
    )
    assert json.loads(out)["input"] == {"mode": "document", "max_words": 5}


def test_predict_text_as_json(capsys: pytest.CaptureFixture[str], baseline_dir: Path) -> None:
    text = "shocking truth exposed secret they hide share wake up bombshell leaked hoax"
    code, out, _ = run_cli(
        capsys, "predict", "--model", str(baseline_dir), "--text", text, "--json"
    )
    assert code == 0
    result = json.loads(out)
    assert result["model"]["backend"] == "baseline"
    assert result["prediction"]["label"] == "FAKE"
    assert 0.5 <= result["prediction"]["p_fake"] <= 1


def test_predict_subtitle_file(capsys: pytest.CaptureFixture[str], baseline_dir: Path) -> None:
    code, out, _ = run_cli(
        capsys, "predict", str(FIXTURES / "sample.srt"), "--model", str(baseline_dir)
    )
    assert code == 0
    assert "P(fake) =" in out
    assert "input: 12 words" in out


def test_predict_long_text_shows_chunks(
    capsys: pytest.CaptureFixture[str], baseline_dir: Path
) -> None:
    text = "officials said the committee report " * 100
    code, out, _ = run_cli(capsys, "predict", "--model", str(baseline_dir), "--text", text)
    assert code == 0
    assert out.startswith("REAL")
    assert "mean over 2 chunks" in out
    assert "preview" in out


def test_predict_reads_stdin(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, baseline_dir: Path
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("officials said the budget report"))
    code, out, _ = run_cli(capsys, "predict", "-", "--model", str(baseline_dir), "--json")
    assert code == 0
    assert json.loads(out)["prediction"]["n_words"] == 5


@pytest.mark.parametrize(
    "argv",
    [
        ["predict"],
        ["predict", "file.txt", "--text", "both"],
        ["predict", "--text", "x", "--threshold", "2"],
        ["predict", "--text", "x", "--chunk-words", "0"],
        ["predict", "--text", "x", "--chunk-words", "50"],
        ["evaluate", "--overlap", "300"],
        ["train", "baseline", "--val-size", "1.5"],
        ["transcribe", VIDEO_URL, "--captions", "--translate"],
        ["run"],
    ],
)
def test_usage_errors(capsys: pytest.CaptureFixture[str], argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_runtime_errors_are_reported_without_traceback(
    capsys: pytest.CaptureFixture[str], baseline_dir: Path, tmp_path: Path
) -> None:
    code, _, err = run_cli(capsys, "predict", "--model", str(tmp_path / "nope"), "--text", "x")
    assert code == 1
    assert err.startswith("error: model directory not found")

    code, _, err = run_cli(
        capsys, "predict", str(tmp_path / "missing.txt"), "--model", str(baseline_dir)
    )
    assert (code, err.strip()) == (1, f"error: file not found: {tmp_path / 'missing.txt'}")

    binary = tmp_path / "audio.txt"
    binary.write_bytes(b"\xff\xfe\x00\x81 not text")
    code, _, err = run_cli(capsys, "predict", str(binary), "--model", str(baseline_dir))
    assert (code, "is not UTF-8 text" in err) == (1, True)

    code, _, err = run_cli(capsys, "predict", "--model", str(baseline_dir), "--text", "[Music]")
    assert code == 1
    assert "empty after cleaning" in err

    code, _, err = run_cli(capsys, "train", "baseline", "--data", str(tmp_path / "none.zip"))
    assert code == 1
    assert "dataset not found" in err


# -------------------------------------------------------- transcribe and run (mocked)


def test_transcribe(
    capsys: pytest.CaptureFixture[str], fake_youtube: FakeYouTube, tmp_path: Path
) -> None:
    code, out, _ = run_cli(capsys, "transcribe", VIDEO_URL, "-o", str(tmp_path), "--device", "cpu")
    assert code == 0
    assert out.startswith("Transcript: 2 segments, 10 words, language en (faster-whisper small)")
    assert (tmp_path / f"{VIDEO['id']}.srt").as_posix() in out
    assert fake_youtube.whisper_loads[0]["device"] == "cpu"


def test_transcribe_captions(
    capsys: pytest.CaptureFixture[str], fake_youtube: FakeYouTube, tmp_path: Path
) -> None:
    code, out, _ = run_cli(capsys, "transcribe", VIDEO_URL, "--captions", "-o", str(tmp_path))
    assert code == 0
    assert "(automatic captions, track en)" in out
    assert fake_youtube.whisper_loads == []


def test_run_prints_video_transcript_and_prediction(
    capsys: pytest.CaptureFixture[str],
    fake_youtube: FakeYouTube,
    baseline_dir: Path,
    tmp_path: Path,
) -> None:
    code, out, _ = run_cli(
        capsys, "run", VIDEO_URL, "--model", str(baseline_dir), "-o", str(tmp_path)
    )
    assert code == 0
    lines = out.splitlines()
    assert lines[0] == "Video:      Evening news"
    assert lines[1].strip() == VIDEO_URL
    assert lines[2].startswith("Transcript: 2 segments")
    assert lines[4].startswith("REAL  P(fake) = ")


def test_run_json(
    capsys: pytest.CaptureFixture[str],
    fake_youtube: FakeYouTube,
    baseline_dir: Path,
    tmp_path: Path,
) -> None:
    code, out, _ = run_cli(
        capsys, "run", VIDEO_URL, "--model", str(baseline_dir), "-o", str(tmp_path), "--json"
    )
    assert code == 0
    result = json.loads(out)
    assert result["video"]["id"] == VIDEO["id"]
    assert result["transcript"]["source"] == "whisper"
    assert result["transcript"]["n_segments"] == 2
    assert Path(result["transcript"]["files"]["txt"]).is_file()
    assert result["prediction"]["label"] == "REAL"
    assert result["prediction"]["n_chunks"] == 1


def test_run_with_no_speech(
    capsys: pytest.CaptureFixture[str],
    fake_youtube: FakeYouTube,
    baseline_dir: Path,
    tmp_path: Path,
) -> None:
    fake_youtube.segments = []
    code, _, err = run_cli(
        capsys, "run", VIDEO_URL, "--model", str(baseline_dir), "-o", str(tmp_path)
    )
    assert code == 1
    assert "is empty (no speech found)" in err


def test_run_checks_the_model_before_downloading(
    capsys: pytest.CaptureFixture[str], fake_youtube: FakeYouTube, tmp_path: Path
) -> None:
    code, _, err = run_cli(capsys, "run", VIDEO_URL, "--model", str(tmp_path / "none"))
    assert code == 1
    assert "model directory not found" in err
    assert fake_youtube.extract_calls == []


def test_missing_asr_extra_is_reported(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    baseline_dir: Path,
    tmp_path: Path,
) -> None:
    monkeypatch.setitem(sys.modules, "yt_dlp", None)
    code, _, err = run_cli(
        capsys, "run", VIDEO_URL, "--model", str(baseline_dir), "-o", str(tmp_path)
    )
    assert code == 1
    assert "error: this feature needs the optional 'asr' dependencies" in err
    assert 'pip install "ytfakenews[asr] @ git+https://github.com/' in err


@pytest.mark.parametrize("blocked", ["torch", "transformers"])
def test_missing_transformer_extra_is_reported(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    news_zip: Path,
    tmp_path: Path,
    blocked: str,
) -> None:
    monkeypatch.setitem(sys.modules, blocked, None)
    out_dir = tmp_path / "model"
    code, _, err = run_cli(
        capsys, "train", "transformer", "--data", str(news_zip), "--output-dir", str(out_dir)
    )
    assert code == 1
    assert "needs the optional 'transformer' dependencies" in err
    assert not out_dir.exists()

    write_manifest(tmp_path, backend="transformer", config={}, data={})
    code, _, err = run_cli(capsys, "predict", "--model", str(tmp_path), "--text", "some words")
    assert code == 1
    assert 'pip install -e ".[transformer]"' in err
