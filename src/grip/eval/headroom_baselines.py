from __future__ import annotations

from typing import Final

from .headroom_types import BaselineSpec


_BASELINE_MODES: Final[tuple[tuple[str, str | None], ...]] = (
    ("dense", None),
    ("local", "local"),
    ("content-sparse", "content_sparse"),
    ("grip-read-A", "grip_read"),
    ("grip-select-B", "grip_select"),
)


BASELINE_NAMES: Final[tuple[str, ...]] = tuple(name for name, _mode in _BASELINE_MODES)


def baseline_specs(read_budget: int) -> tuple[BaselineSpec, ...]:
    return tuple(
        BaselineSpec(
            name=name,
            attention_mode=attention_mode,
            read_budget=None if attention_mode is None else read_budget,
        )
        for name, attention_mode in _BASELINE_MODES
    )
