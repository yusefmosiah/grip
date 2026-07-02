"""Minimal training artifact writer.

The trainer's job: produce artifacts (checkpoint, JSONL log, eval tensors).
The trainer does NOT decide who won — that's eval/score.py's job.

Accepts a config dict or JSON config path. Configs are reproducibility: same
config + same seed => same run.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
from typing import Mapping, Sequence, TypeAlias

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | Mapping[str, JsonScalar]


@dataclass(frozen=True, slots=True)
class ConfigError(ValueError):
    section: str
    field: str
    reason: str

    def __str__(self) -> str:
        return f"{self.section}.{self.field}: {self.reason}"


@dataclass(frozen=True, slots=True)
class ModelConfig:
    name: str
    size: str
    d_model: int
    n_layers: int
    top_k_blocks: int | None


@dataclass(frozen=True, slots=True)
class DataConfig:
    task: str
    seq_len: int
    num_hypotheses: int


@dataclass(frozen=True, slots=True)
class TrainConfig:
    steps: int
    microbatch: int
    grad_accum: int
    lr: float
    seed: int


@dataclass(frozen=True, slots=True)
class RunSettings:
    mode: str


@dataclass(frozen=True, slots=True)
class RunConfig:
    model: ModelConfig
    data: DataConfig
    train: TrainConfig
    device: str
    run: RunSettings


def train(config: RunConfig | Mapping[str, JsonValue], run_dir: str | Path = "runs/default") -> Path:
    """Run one artifact-plumbing job. Writes resolved config, JSONL log, and metrics.

    config keys (minimal):
        model: {name: 'dense'|'sparse'|'grip-read-A'|'grip-select-B', ...}
        data:  {task:'bayesian', seq_len, num_hypotheses, ...}
        train: {steps, microbatch, grad_accum, lr, precision, seed}
        device: 'mps' | 'cpu'
        run: {mode: 'stub-dry-run'}
    """
    resolved = parse_run_config(_config_mapping(config) if isinstance(config, RunConfig) else config)
    if resolved.run.mode != "stub-dry-run":
        raise ConfigError("run", "mode", "stub trainer requires explicit stub-dry-run mode")
    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_payload = _resolved_payload(resolved)
    (out_dir / "config.resolved.json").write_text(
        json.dumps(resolved_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    train_record = {
        "device": resolved.device,
        "event": "dry_run",
        "elapsed_seconds": 0.0,
        "loss": {"total": 0.0},
        "lr": resolved.train.lr,
        "mode": resolved.run.mode,
        "step": 0,
        "tokens": 0,
    }
    (out_dir / "train.jsonl").write_text(
        json.dumps(train_record, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "eval_tensors.json").write_text(
        json.dumps({"loss": 0.0, "tokens": 0.0}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_dir


def load_run_config(path: Path) -> Mapping[str, JsonValue]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError("root", "config", "must be valid JSON") from exc
    if not isinstance(raw, Mapping):
        raise ConfigError("root", "config", "must be a JSON object")
    return raw


def parse_run_config(config: Mapping[str, JsonValue]) -> RunConfig:
    model = _parse_model(_require_mapping(config, "model"))
    data = _parse_data(_require_mapping(config, "data"))
    train_cfg = _parse_train(_require_mapping(config, "train"))
    device = _require_str(config, "root", "device")
    if device not in {"cpu", "mps"}:
        raise ConfigError("root", "device", "must be cpu or mps")
    run = _parse_run(_require_mapping(config, "run"))
    return RunConfig(model=model, data=data, train=train_cfg, device=device, run=run)


def _config_mapping(config: RunConfig) -> Mapping[str, JsonValue]:
    return {
        "data": asdict(config.data),
        "device": config.device,
        "model": asdict(config.model),
        "run": asdict(config.run),
        "train": asdict(config.train),
    }


def _parse_model(raw: Mapping[str, JsonScalar]) -> ModelConfig:
    name = _require_str(raw, "model", "name")
    if name not in {"dense", "sparse", "grip-read-A", "grip-select-B"}:
        raise ConfigError("model", "name", "must be dense, sparse, grip-read-A, or grip-select-B")
    top_k_blocks = _optional_int(raw, "model", "top_k_blocks")
    if name != "dense" and top_k_blocks is None:
        raise ConfigError("model", "top_k_blocks", "required for sparse variants")
    return ModelConfig(
        name=name,
        size=_require_str(raw, "model", "size"),
        d_model=_require_int(raw, "model", "d_model"),
        n_layers=_require_int(raw, "model", "n_layers"),
        top_k_blocks=top_k_blocks,
    )


def _parse_data(raw: Mapping[str, JsonScalar]) -> DataConfig:
    return DataConfig(
        task=_require_str(raw, "data", "task"),
        seq_len=_require_int(raw, "data", "seq_len"),
        num_hypotheses=_require_int(raw, "data", "num_hypotheses"),
    )


def _parse_train(raw: Mapping[str, JsonScalar]) -> TrainConfig:
    return TrainConfig(
        steps=_require_int(raw, "train", "steps"),
        microbatch=_require_int(raw, "train", "microbatch"),
        grad_accum=_require_int(raw, "train", "grad_accum"),
        lr=_require_float(raw, "train", "lr"),
        seed=_require_int(raw, "train", "seed"),
    )


def _parse_run(raw: Mapping[str, JsonScalar]) -> RunSettings:
    mode = _require_str(raw, "run", "mode")
    if mode not in {"stub-dry-run", "smoke", "preregistered"}:
        raise ConfigError("run", "mode", "must be stub-dry-run, smoke, or preregistered")
    return RunSettings(mode=mode)


def _resolved_payload(config: RunConfig) -> Mapping[str, JsonValue]:
    return {
        "artifact_schema_version": 1,
        "data": asdict(config.data),
        "device": config.device,
        "git_sha": _git_sha(),
        "model": asdict(config.model),
        "read_budget": config.model.top_k_blocks,
        "run": asdict(config.run),
        "train": asdict(config.train),
    }


def _require_mapping(config: Mapping[str, JsonValue], section: str) -> Mapping[str, JsonScalar]:
    raw = config.get(section)
    if not isinstance(raw, Mapping):
        raise ConfigError("root", section, "section is required")
    return raw


def _require_str(raw: Mapping[str, JsonScalar], section: str, field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value:
        raise ConfigError(section, field, "non-empty string is required")
    return value


def _require_int(raw: Mapping[str, JsonScalar], section: str, field: str) -> int:
    value = raw.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(section, field, "integer is required")
    return value


def _optional_int(raw: Mapping[str, JsonScalar], section: str, field: str) -> int | None:
    value = raw.get(field)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(section, field, "integer is required")
    return value


def _require_float(raw: Mapping[str, JsonScalar], section: str, field: str) -> float:
    value = raw.get(field)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ConfigError(section, field, "number is required")
    return float(value)


def _git_sha() -> str:
    root = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    run_dir = train(load_run_config(args.config), args.run_dir)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
