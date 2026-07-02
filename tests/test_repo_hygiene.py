from __future__ import annotations

from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize(
    "relative_path",
    (
        ".omo/scratch-check.tmp",
        "runs/smoke/metrics.json",
        "checkpoints/metadata.json",
        "models/tiny.pt",
    ),
)
def test_generated_artifacts_are_gitignored(relative_path: str) -> None:
    root = Path(__file__).resolve().parents[1]
    scratch = root / relative_path

    result = subprocess.run(
        ["git", "check-ignore", str(scratch)],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
