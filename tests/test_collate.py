from __future__ import annotations

from grip.data.collate import _sample_seed


def test_sample_seed_avoids_linear_batch_seed_collision() -> None:
    # Given: the historical collision pair from the old seed * 1000 + index scheme.
    first = _sample_seed(1, 0)
    second = _sample_seed(0, 1000)

    # Then: sample seed derivation keeps them distinct.
    assert first != second
