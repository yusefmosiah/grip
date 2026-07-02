from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import TypedDict

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


class StreamPayload(TypedDict):
    name: str
    seq_len: int
    num_hypotheses: int
    num_sources: int
    vocab_size: int
    stream_seed: int


class MetricsPayload(TypedDict):
    d_conf_mse: float
    d_conf_r2: float
    answer_accuracy: float
    positive_control_r2: float
    answer_train_loss_initial: float
    answer_train_loss_final: float


class ThresholdPayload(TypedDict):
    d_conf_r2_threshold: float
    answer_acc_threshold: float
    positive_control_r2_threshold: float


class DecisionPayload(TypedDict):
    d_conf_passed: bool
    answer_passed: bool
    positive_control_passed: bool
    answer_converged: bool
    passed: bool


class SelectedProbePayload(TypedDict):
    window: int
    ridge: float
    answer_window: int
    window_grid: list[int]
    ridge_grid: list[float]


class TaskReport(TypedDict):
    gate: str
    task: str
    stream: StreamPayload
    probe_config: dict[str, object]
    metrics: MetricsPayload
    thresholds: ThresholdPayload
    decision: DecisionPayload
    selected_probe: SelectedProbePayload


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
            "positive_control_r2": result.positive_control_r2,
            "answer_train_loss_initial": result.answer_train_loss_initial,
            "answer_train_loss_final": result.answer_train_loss_final,
        },
        "thresholds": {
            "d_conf_r2_threshold": result.d_conf_r2_threshold,
            "answer_acc_threshold": result.answer_acc_threshold,
            "positive_control_r2_threshold": result.positive_control_r2_threshold,
        },
        "decision": {
            "d_conf_passed": result.d_conf_passed,
            "answer_passed": result.answer_passed,
            "positive_control_passed": result.positive_control_passed,
            "answer_converged": result.answer_converged,
            "passed": result.passed,
        },
        "selected_probe": {
            "window": result.window,
            "ridge": result.ridge,
            "answer_window": result.answer_window,
            "window_grid": list(result.window_grid),
            "ridge_grid": list(result.ridge_grid),
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
            "positive_control_passed": all(
                bool(report["decision"]["positive_control_passed"])
                for report in reports.values()
            ),
            "answer_converged": all(
                bool(report["decision"]["answer_converged"])
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
