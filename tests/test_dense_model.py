"""Dense model tests: shapes, MPS execution, and the overfit-one-batch gate.

The overfit test is milestone M-overfit: if a tiny model cannot drive loss to
~0 on a single fixed batch, the wiring is wrong — stop before anything else.
"""
import pytest
import torch
import torch.nn.functional as F

from grip.models import DenseTransformer
from grip.data import BayesianEvidenceStream, make_batch


def _tiny_model(stream):
    return DenseTransformer(
        vocab_size=stream.vocab_size,
        d_model=64, n_heads=4, n_layers=2,
        max_seq_len=stream.T, n_hypotheses=stream.K,
    )


def test_forward_shapes():
    stream = BayesianEvidenceStream(num_hypotheses=4, seq_len=128, vocab_size=32, seed=0)
    model = _tiny_model(stream)
    batch = make_batch(stream, n=4)
    out = model(batch["tokens"])
    B, T = batch["tokens"].shape
    assert out["lm_logits"].shape == (B, T, stream.vocab_size)
    assert out["posterior"].shape == (B, T, stream.K)
    assert out["hidden"].shape == (B, T, 64)


def test_posterior_is_simplex():
    stream = BayesianEvidenceStream(num_hypotheses=4, seq_len=64, vocab_size=32, seed=0)
    model = _tiny_model(stream)
    batch = make_batch(stream, n=2)
    post = model(batch["tokens"])["posterior"]
    assert torch.allclose(post.sum(-1), torch.ones(post.shape[:-1]), atol=1e-5)


def test_runs_on_mps_if_available():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    stream = BayesianEvidenceStream(num_hypotheses=4, seq_len=128, vocab_size=32, seed=0)
    model = _tiny_model(stream).to("mps")
    batch = make_batch(stream, n=2, device="mps")
    out = model(batch["tokens"])
    assert out["hidden"].device.type == "mps"
    assert torch.isfinite(out["lm_logits"]).all()


def test_overfit_one_batch():
    """M-overfit: a tiny model must drive loss toward 0 on ONE fixed batch.
    This is the wiring gate — if it fails, stop."""
    torch.manual_seed(0)
    stream = BayesianEvidenceStream(num_hypotheses=3, seq_len=64, vocab_size=16, seed=0)
    model = _tiny_model(stream)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    batch = make_batch(stream, n=4, seed=0)

    # loss = next-token LM loss + auxiliary posterior loss (matches SPEC-000 training)
    target_tokens = batch["tokens"]
    target_post = batch["posterior"]
    initial_loss = None
    for step in range(400):
        out = model(batch["tokens"])
        lm = F.cross_entropy(
            out["lm_logits"][:, :-1].reshape(-1, stream.vocab_size),
            target_tokens[:, 1:].reshape(-1),
        )
        # KL from model posterior to ground-truth posterior (aux supervision)
        log_p = torch.log(out["posterior"] + 1e-8)
        aux = (target_post * (torch.log(target_post + 1e-8) - log_p)).sum(-1).mean()
        loss = lm + 0.1 * aux
        if initial_loss is None:
            initial_loss = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    final_loss = loss.item()
    assert final_loss < initial_loss * 0.3, (
        f"model failed to overfit one batch: {initial_loss:.3f} -> {final_loss:.3f}"
    )
