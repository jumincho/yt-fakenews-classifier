"""Keep the notebook, the examples and the READMEs consistent with the package."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pytest

from ytfakenews.cli import build_parser

ROOT = Path(__file__).parents[1]
NOTEBOOK = ROOT / "notebooks" / "colab_quickstart.ipynb"


def _code_cells() -> list[str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    for cell in cells:
        assert cell["outputs"] == [], f"cell {cell['id']} has saved outputs"
        assert cell["execution_count"] is None
    return ["".join(cell["source"]) for cell in cells]


def test_notebook_has_no_outputs_and_valid_python() -> None:
    for index, source in enumerate(_code_cells()):
        python = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith(("!", "%"))
        )
        compile(python, f"{NOTEBOOK.name} code cell {index}", "exec")


def test_notebook_only_uses_existing_commands(capsys: pytest.CaptureFixture[str]) -> None:
    used = set(re.findall(r"(?<!from )ytfakenews (\w+)", "\n".join(_code_cells())))
    assert used == {"train", "run", "predict"}
    for command in used:
        with pytest.raises(SystemExit) as exit_info:
            build_parser().parse_args([command, "--help"])
        assert exit_info.value.code == 0, command
    capsys.readouterr()


@pytest.mark.parametrize("name", ["transcript_local_news.txt", "transcript_sensational.txt"])
def test_example_transcripts_exist(name: str) -> None:
    text = (ROOT / "examples" / name).read_text(encoding="utf-8")
    assert 200 < len(text.split()) < 400


READMES = ["README.md", "README.zh-CN.md", "README.zh-HK.md", "README.ja.md", "README.ko.md"]
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_HEADING = re.compile(r"^ {0,3}#{1,6}[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$")


def _prose_lines(markdown: str) -> list[str]:
    """Return the lines of ``markdown`` that are outside fenced code blocks."""
    lines: list[str] = []
    fence = ""
    for line in markdown.splitlines():
        match = _FENCE.match(line)
        if not fence:
            if match:
                fence = match.group(1)
            else:
                lines.append(line)
        elif (
            match
            and match.group(1)[0] == fence[0]
            and len(match.group(1)) >= len(fence)
            and not match.group(2).strip()
        ):
            fence = ""
    return lines


def _github_anchor(heading: str) -> str:
    """Return GitHub's anchor for a heading.

    GitHub takes the heading's text without Markdown syntax, lower-cases it, deletes every
    character that is not a letter, mark, number, connector punctuation (such as '_'), space
    or hyphen, and turns each space into '-'. Kana, CJK ideographs and Hangul are letters and
    stay, while ASCII and full-width punctuation is deleted.
    """
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)  # a link counts as its text
    kept = (
        char
        for char in text.strip().lower()
        if char in " -" or unicodedata.category(char).startswith(("L", "M", "N", "Pc"))
    )
    return "".join(kept).replace(" ", "-")


def _anchors(markdown: str) -> set[str]:
    """Return the anchors of the headings in ``markdown``; repeats get -1, -2, ..."""
    anchors: set[str] = set()
    seen: Counter[str] = Counter()
    for line in _prose_lines(markdown):
        if match := _HEADING.match(line):
            anchor = _github_anchor(match.group(1))
            anchors.add(f"{anchor}-{seen[anchor]}" if seen[anchor] else anchor)
            seen[anchor] += 1
    return anchors


@pytest.mark.parametrize(
    ("heading", "anchor"),
    [
        ("Model card and limitations", "model-card-and-limitations"),
        ("`ytfakenews` **CLI** [reference](#cli)", "ytfakenews-cli-reference"),
        ("模型卡与局限性", "模型卡与局限性"),
        ("モデルカードと制限事項", "モデルカードと制限事項"),
        ("모델 카드와 한계", "모델-카드와-한계"),
        ("1. 시도별 당선 정당", "1-시도별-당선-정당"),
        (
            "安裝\N{FULLWIDTH LEFT PARENTHESIS}可選\N{FULLWIDTH RIGHT PARENTHESIS}"
            "\N{FULLWIDTH COLON}步驟、說明。為何\N{FULLWIDTH QUESTION MARK}",
            "安裝可選步驟說明為何",
        ),
    ],
)
def test_github_anchor(heading: str, anchor: str) -> None:
    assert _github_anchor(heading) == anchor


def test_anchors_skip_code_blocks_and_number_repeats() -> None:
    markdown = "# 概要\n\n```bash\n# 这是注释\n```\n\n~~~text\n## Not a heading\n~~~\n\n## 概要\n"
    assert _anchors(markdown) == {"概要", "概要-1"}


@pytest.mark.parametrize("name", READMES)
def test_readme_links_resolve(name: str) -> None:
    readme = (ROOT / name).read_text(encoding="utf-8")
    targets = re.findall(r"\]\(([^)\s]+)\)", "\n".join(_prose_lines(readme)))
    anchors = _anchors(readme)
    local = [target for target in targets if not re.match(r"[a-z][a-z0-9+.-]*:", target)]
    assert local
    for target in local:
        path, _, anchor = target.partition("#")
        if path:
            assert (ROOT / path).exists(), f"{name} links to missing {path}"
        else:
            assert anchor in anchors, f"{name} links to missing section #{anchor}"
