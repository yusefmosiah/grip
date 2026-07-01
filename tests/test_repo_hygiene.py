from __future__ import annotations

from pathlib import Path
import subprocess


def test_omo_research_artifacts_are_gitignored():
    # Given: agent/research scratch artifacts live under .omo/.
    root = Path(__file__).resolve().parents[1]
    scratch = root / ".omo" / "scratch-check.tmp"

    # When / Then: Git treats the scratch path as ignored behaviorally.
    result = subprocess.run(
        ["git", "check-ignore", str(scratch)],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
