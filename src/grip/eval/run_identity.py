from __future__ import annotations

from typing import Mapping, Sequence

from .score import _field_value, _load_run_config
from .score_types import RunScore, ScoreArtifactError


def run_model_name(score: RunScore) -> str:
    config_path = score.run_dir / "config.resolved.json"
    run_config = _load_run_config(config_path)
    name = _field_value(run_config, ("model", "name"))
    if not isinstance(name, str) or not name:
        raise ScoreArtifactError(config_path, "model.name must be a non-empty string")
    return name


def scores_by_model_name(scores: Sequence[RunScore]) -> Mapping[str, RunScore]:
    by_name: dict[str, RunScore] = {}
    for score in scores:
        name = run_model_name(score)
        if name in by_name:
            raise ScoreArtifactError(
                score.run_dir / "config.resolved.json",
                f"duplicate model.name {name!r}",
            )
        by_name[name] = score
    return by_name
