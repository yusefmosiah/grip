from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

from grip.analysis.bypass import BypassProbeConfig, run_bypass_probe
from grip.data import BayesianEvidenceStream


@dataclass(frozen=True, slots=True)
class BypassRunConfig:
    out_dir: str = "runs/bypass-002"
    seq_len: int = 128
    num_hypotheses: int = 4
    num_sources: int = 3
    vocab_size: int = 64
    stream_seed: int = 0
    device: str = "cpu"
    probe: BypassProbeConfig = field(default_factory=BypassProbeConfig)


def run_gate(config: BypassRunConfig) -> dict[str, int | float | bool | str | dict[str, int | float]]:
    stream = BayesianEvidenceStream(
        num_hypotheses=config.num_hypotheses,
        num_sources=config.num_sources,
        seq_len=config.seq_len,
        vocab_size=config.vocab_size,
        seed=config.stream_seed,
    )
    result = run_bypass_probe(stream, config.probe, device=config.device)
    report = {
        "gate": "M-legibility",
        "stream": {
            "seq_len": config.seq_len,
            "num_hypotheses": config.num_hypotheses,
            "num_sources": config.num_sources,
            "vocab_size": config.vocab_size,
            "stream_seed": config.stream_seed,
        },
        "probe_config": asdict(config.probe),
        "metrics": {
            "d_conf_mse": result.d_conf_mse,
            "d_conf_r2": result.d_conf_r2,
            "answer_accuracy": result.answer_accuracy,
        },
        "thresholds": {
            "d_conf_r2_threshold": result.d_conf_r2_threshold,
            "answer_acc_threshold": result.answer_acc_threshold,
        },
        "decision": {
            "d_conf_passed": result.d_conf_passed,
            "answer_passed": result.answer_passed,
            "passed": result.passed,
        },
    }
    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    report = run_gate(BypassRunConfig())
    decision = report["decision"]
    metrics = report["metrics"]
    print("M-legibility bypass gate")
    print(f"  d_conf R^2: {metrics['d_conf_r2']:+.4f}")
    print(f"  answer accuracy: {metrics['answer_accuracy']:.4f}")
    print(f"  passed: {decision['passed']}")


if __name__ == "__main__":
    main()
