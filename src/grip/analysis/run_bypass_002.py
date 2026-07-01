from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

from grip.analysis.bypass import BypassProbeConfig, run_bypass_probe
from grip.data import BayesianEvidenceStream, SourceReliabilityReversalStream


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


TaskReport = dict[str, int | float | bool | str | dict[str, int | float]]


def _t0_stream(config: BypassRunConfig) -> BayesianEvidenceStream:
    return BayesianEvidenceStream(
        num_hypotheses=config.num_hypotheses,
        num_sources=config.num_sources,
        seq_len=config.seq_len,
        vocab_size=config.vocab_size,
        seed=config.stream_seed,
    )


def _t1_stream(config: BypassRunConfig) -> SourceReliabilityReversalStream:
    return SourceReliabilityReversalStream(
        seq_len=config.seq_len,
        seed=config.stream_seed,
    )


def _task_report(
    task_name: str,
    stream_name: str,
    stream: BayesianEvidenceStream | SourceReliabilityReversalStream,
    config: BypassRunConfig,
) -> TaskReport:
    result = run_bypass_probe(stream, config.probe, device=config.device)
    return {
        "gate": "M-legibility",
        "task": task_name,
        "stream": {
            "name": stream_name,
            "seq_len": config.seq_len,
            "num_hypotheses": stream.K,
            "num_sources": stream.S,
            "vocab_size": stream.vocab_size,
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


def run_gate(config: BypassRunConfig) -> dict[str, object]:
    reports = {
        "T0-bayesian-evidence-streams": _task_report(
            "T0-bayesian-evidence-streams",
            "BayesianEvidenceStream",
            _t0_stream(config),
            config,
        ),
        "T1-source-reliability-reversal": _task_report(
            "T1-source-reliability-reversal",
            "SourceReliabilityReversalStream",
            _t1_stream(config),
            config,
        ),
    }
    aggregate = {
        "gate": "M-legibility",
        "tasks": tuple(reports),
        "reports": reports,
        "probe_config": asdict(config.probe),
        "metrics": {
            task_name: report["metrics"]
            for task_name, report in reports.items()
        },
        "thresholds": reports["T0-bayesian-evidence-streams"]["thresholds"],
        "decision": {
            "d_conf_passed": all(
                bool(report["decision"]["d_conf_passed"])
                for report in reports.values()
            ),
            "answer_passed": all(
                bool(report["decision"]["answer_passed"])
                for report in reports.values()
            ),
            "passed": all(
                bool(report["decision"]["passed"])
                for report in reports.values()
            ),
        },
    }
    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for task_name, report in reports.items():
        (out / f"{task_name}-report.json").write_text(json.dumps(report, indent=2))
    (out / "report.json").write_text(json.dumps(aggregate, indent=2))
    return aggregate


def main() -> None:
    report = run_gate(BypassRunConfig())
    print("M-legibility bypass gate")
    for task_name, task_report in report["reports"].items():
        decision = task_report["decision"]
        metrics = task_report["metrics"]
        print(f"  {task_name}")
        print(f"    d_conf R^2: {metrics['d_conf_r2']:+.4f}")
        print(f"    answer accuracy: {metrics['answer_accuracy']:.4f}")
        print(f"    passed: {decision['passed']}")
    print(f"  aggregate passed: {report['decision']['passed']}")


if __name__ == "__main__":
    main()
