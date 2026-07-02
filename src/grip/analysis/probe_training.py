from __future__ import annotations

from dataclasses import dataclass
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from grip.data import make_batch
from grip.models import DenseTransformer


BACKBONE_BATCH_SEED_BASE = 1_000_000_000
BACKBONE_BATCH_SEED_STRIDE = 1_000_000
BACKBONE_SEED_LIMIT = 9_000


@dataclass(frozen=True, slots=True)
class SupervisionWeights:
    lm: float = 0.1
    posterior: float = 5.0
    topmass: float = 10.0
    entropy: float = 3.0
    d_conf: float = 0.0
    dd_conf: float = 0.0

    @property
    def derivative_enabled(self) -> bool:
        return self.d_conf > 0.0 or self.dd_conf > 0.0

    def as_report(self) -> dict[str, float | bool]:
        return {
            "lm_weight": self.lm,
            "aux_weight": self.posterior,
            "topmass_weight": self.topmass,
            "entropy_weight": self.entropy,
            "d_conf_weight": self.d_conf,
            "dd_conf_weight": self.dd_conf,
            "derivative_supervision_enabled": self.derivative_enabled,
        }


@dataclass(frozen=True, slots=True)
class AuxiliaryLosses:
    topmass: float
    entropy: float
    d_conf: float
    dd_conf: float

    def as_report(self) -> dict[str, float]:
        return {
            "topmass_loss": self.topmass,
            "entropy_loss": self.entropy,
            "d_conf_loss": self.d_conf,
            "dd_conf_loss": self.dd_conf,
        }


@dataclass(slots=True)
class AuxiliaryHeads:
    topmass: nn.Linear
    entropy: nn.Linear
    d_conf: nn.Linear
    dd_conf: nn.Linear


@dataclass(frozen=True, slots=True)
class TrainingResult:
    model: DenseTransformer
    final_losses: AuxiliaryLosses


def backbone_batch_seed_base(seed: int) -> int:
    if seed < 0 or seed >= BACKBONE_SEED_LIMIT:
        msg = f"seed must be in [0, {BACKBONE_SEED_LIMIT}) for disjoint backbone namespace"
        raise ValueError(msg)
    return BACKBONE_BATCH_SEED_BASE + seed * BACKBONE_BATCH_SEED_STRIDE


def _masked_scalar_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred[mask], target[mask])


def _auxiliary_loss(
    hidden: torch.Tensor,
    batch_data: dict[str, torch.Tensor],
    heads: AuxiliaryHeads,
    weights: SupervisionWeights,
) -> tuple[torch.Tensor, AuxiliaryLosses]:
    real_mask = batch_data["real_mask"]
    topmass = batch_data["posterior"].max(-1).values
    topmass_loss = _masked_scalar_mse(heads.topmass(hidden).squeeze(-1), topmass, real_mask)
    entropy_loss = _masked_scalar_mse(heads.entropy(hidden).squeeze(-1), batch_data["entropy"], real_mask)
    d_conf_loss = _masked_scalar_mse(heads.d_conf(hidden).squeeze(-1), batch_data["d_conf"], real_mask)
    dd_conf_loss = _masked_scalar_mse(heads.dd_conf(hidden).squeeze(-1), batch_data["dd_conf"], real_mask)
    loss = (
        weights.topmass * topmass_loss
        + weights.entropy * entropy_loss
        + weights.d_conf * d_conf_loss
        + weights.dd_conf * dd_conf_loss
    )
    components = AuxiliaryLosses(
        topmass=float(topmass_loss.detach().cpu().item()),
        entropy=float(entropy_loss.detach().cpu().item()),
        d_conf=float(d_conf_loss.detach().cpu().item()),
        dd_conf=float(dd_conf_loss.detach().cpu().item()),
    )
    return loss, components


def train_backbone(
    stream, d_model=128, n_layers=4, n_heads=4,
    n_steps=1500, batch=8, lr=5e-4, device="mps", seed=0,
    lm_weight=0.1, aux_weight=5.0, topmass_weight=10.0,
    entropy_weight=3.0, supervision: SupervisionWeights | None = None,
    log_every=100,
) -> DenseTransformer:
    return train_backbone_with_report(
        stream,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        n_steps=n_steps,
        batch=batch,
        lr=lr,
        device=device,
        seed=seed,
        lm_weight=lm_weight,
        aux_weight=aux_weight,
        topmass_weight=topmass_weight,
        entropy_weight=entropy_weight,
        supervision=supervision,
        log_every=log_every,
    ).model


def train_backbone_with_report(
    stream, d_model=128, n_layers=4, n_heads=4,
    n_steps=1500, batch=8, lr=5e-4, device="mps", seed=0,
    lm_weight=0.1, aux_weight=5.0, topmass_weight=10.0,
    entropy_weight=3.0, supervision: SupervisionWeights | None = None,
    log_every=100,
) -> TrainingResult:
    weights = supervision or SupervisionWeights(
        lm=lm_weight,
        posterior=aux_weight,
        topmass=topmass_weight,
        entropy=entropy_weight,
    )
    torch.manual_seed(seed)
    model = DenseTransformer(
        vocab_size=stream.vocab_size, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, max_seq_len=stream.T, n_hypotheses=stream.K,
    ).to(device)
    heads = AuxiliaryHeads(
        topmass=nn.Linear(d_model, 1).to(device),
        entropy=nn.Linear(d_model, 1).to(device),
        d_conf=nn.Linear(d_model, 1).to(device),
        dd_conf=nn.Linear(d_model, 1).to(device),
    )
    opt = torch.optim.Adam(
        list(model.parameters())
        + list(heads.topmass.parameters())
        + list(heads.entropy.parameters())
        + list(heads.d_conf.parameters())
        + list(heads.dd_conf.parameters()),
        lr=lr,
    )
    seed_seq = backbone_batch_seed_base(seed)
    if n_steps >= BACKBONE_BATCH_SEED_STRIDE:
        msg = "n_steps must fit within the per-backbone seed stride"
        raise ValueError(msg)
    t0 = time.time()
    final_losses = AuxiliaryLosses(topmass=0.0, entropy=0.0, d_conf=0.0, dd_conf=0.0)
    for step in range(n_steps):
        batch_data = make_batch(stream, n=batch, seed=seed_seq + step, device=device)
        out = model(batch_data["tokens"])
        hidden = out.hidden
        lm = F.cross_entropy(
            out.lm_logits[:, :-1].reshape(-1, stream.vocab_size),
            batch_data["tokens"][:, 1:].reshape(-1),
        )
        log_p = torch.log(out.posterior + 1e-8)
        aux = (batch_data["posterior"] * (
            torch.log(batch_data["posterior"] + 1e-8) - log_p)).sum(-1).mean()
        auxiliary_loss, final_losses = _auxiliary_loss(hidden, batch_data, heads, weights)
        loss = (
            weights.lm * lm
            + weights.posterior * aux
            + auxiliary_loss
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % log_every == 0:
            mem = (torch.mps.current_allocated_memory() / 1e6
                   if device == "mps" and hasattr(torch.mps, "current_allocated_memory") else 0)
            print(f"  step {step:4d}  lm={lm.item():.3f}  aux={aux.item():.4f}  "
                  f"top={final_losses.topmass:.4f}  ent={final_losses.entropy:.4f}  "
                  f"dc={final_losses.d_conf:.4f}  ddc={final_losses.dd_conf:.4f}  "
                  f"loss={loss.item():.3f}  mem={mem:.0f}MB  dt={time.time()-t0:.0f}s")
    return TrainingResult(model=model, final_losses=final_losses)
