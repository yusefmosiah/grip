import torch

from grip.analysis.probe_training import AuxiliaryHeads, SupervisionWeights, _auxiliary_loss
from grip.analysis.run_probe_000 import train_backbone
from grip.data import BayesianEvidenceStream
from grip.models import DenseTransformer


def test_train_backbone_accepts_level_auxiliary_weights():
    stream = BayesianEvidenceStream(num_hypotheses=2, num_sources=1, seq_len=8, vocab_size=8, seed=0)

    model = train_backbone(
        stream,
        d_model=16,
        n_layers=1,
        n_heads=4,
        n_steps=1,
        batch=1,
        device="cpu",
        lm_weight=0.1,
        aux_weight=1.0,
        topmass_weight=1.0,
        entropy_weight=1.0,
        log_every=99,
    )

    assert isinstance(model, DenseTransformer)


def test_train_backbone_accepts_derivative_auxiliary_supervision():
    # Given: derivative targets are explicitly supervised alongside level targets.
    stream = BayesianEvidenceStream(num_hypotheses=2, num_sources=1, seq_len=8, vocab_size=8, seed=0)
    supervision = SupervisionWeights(
        lm=0.1,
        posterior=1.0,
        topmass=1.0,
        entropy=1.0,
        d_conf=10.0,
        dd_conf=10.0,
    )

    # When: the backbone trains for one CPU step with derivative supervision enabled.
    model = train_backbone(
        stream,
        d_model=16,
        n_layers=1,
        n_heads=4,
        n_steps=1,
        batch=1,
        device="cpu",
        supervision=supervision,
        log_every=99,
    )

    # Then: the same backbone surface can be probed after derivative-supervised training.
    assert isinstance(model, DenseTransformer)
    assert supervision.derivative_enabled
    assert supervision.as_report()["d_conf_weight"] == 10.0
    assert supervision.as_report()["dd_conf_weight"] == 10.0


def test_derivative_weights_affect_masked_auxiliary_loss():
    # Given: zero heads and nonzero real derivative targets.
    heads = _zero_auxiliary_heads(d_model=3)
    hidden = torch.zeros((1, 2, 3), dtype=torch.float32)
    batch_data = _auxiliary_batch(
        d_conf=torch.tensor([[2.0, 99.0]]),
        dd_conf=torch.tensor([[3.0, 99.0]]),
        real_mask=torch.tensor([[True, False]]),
    )

    # When: derivative weights are enabled instead of left at zero.
    without_derivative, _ = _auxiliary_loss(hidden, batch_data, heads, SupervisionWeights())
    with_derivative, components = _auxiliary_loss(
        hidden,
        batch_data,
        heads,
        SupervisionWeights(topmass=0.0, entropy=0.0, d_conf=5.0, dd_conf=7.0),
    )

    # Then: the real derivative targets contribute to the scalar aux loss.
    assert with_derivative.item() > without_derivative.item()
    assert components.d_conf == 4.0
    assert components.dd_conf == 9.0


def test_auxiliary_loss_masks_padded_derivative_artifacts():
    # Given: derivative targets are zero on real positions and huge only on padding.
    heads = _zero_auxiliary_heads(d_model=3)
    hidden = torch.zeros((1, 2, 3), dtype=torch.float32)
    batch_data = _auxiliary_batch(
        d_conf=torch.tensor([[0.0, 100.0]]),
        dd_conf=torch.tensor([[0.0, -100.0]]),
        real_mask=torch.tensor([[True, False]]),
    )

    # When: derivative supervision is enabled.
    loss, components = _auxiliary_loss(
        hidden,
        batch_data,
        heads,
        SupervisionWeights(topmass=0.0, entropy=0.0, d_conf=5.0, dd_conf=7.0),
    )

    # Then: padded derivative values do not affect the supervised loss.
    assert loss.item() == 0.0
    assert components.d_conf == 0.0
    assert components.dd_conf == 0.0


def _zero_auxiliary_heads(d_model: int) -> AuxiliaryHeads:
    heads = AuxiliaryHeads(
        topmass=torch.nn.Linear(d_model, 1),
        entropy=torch.nn.Linear(d_model, 1),
        d_conf=torch.nn.Linear(d_model, 1),
        dd_conf=torch.nn.Linear(d_model, 1),
    )
    for head in (heads.topmass, heads.entropy, heads.d_conf, heads.dd_conf):
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.zeros_(head.bias)
    return heads


def _auxiliary_batch(
    d_conf: torch.Tensor,
    dd_conf: torch.Tensor,
    real_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "posterior": torch.zeros((*d_conf.shape, 2), dtype=torch.float32),
        "entropy": torch.zeros(d_conf.shape, dtype=torch.float32),
        "d_conf": d_conf,
        "dd_conf": dd_conf,
        "real_mask": real_mask,
    }
