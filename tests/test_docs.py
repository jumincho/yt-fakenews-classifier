"""Keep the notebook and the examples consistent with the package."""

from __future__ import annotations

import json
import re
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


def _github_anchor(heading: str) -> str:
    """Approximate GitHub's heading slugs: lower case, punctuation dropped, spaces to '-'."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def test_readme_links_resolve() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"\]\(([^)\s]+)\)", readme)
    anchors = {_github_anchor(h) for h in re.findall(r"^#+ (.+)$", readme, flags=re.MULTILINE)}
    local = [target for target in targets if not re.match(r"[a-z][a-z0-9+.-]*:", target)]
    assert local
    for target in local:
        path, _, anchor = target.partition("#")
        if path:
            assert (ROOT / path).exists(), f"README links to missing {path}"
        else:
            assert anchor in anchors, f"README links to missing section #{anchor}"
