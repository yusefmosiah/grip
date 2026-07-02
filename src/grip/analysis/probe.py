"""Linear probes for the gating experiment (SPEC-000).

The gating question: is the certainty-TRAJECTORY (d_conf, dd_conf) already
present in the hidden state of a vanilla model, or is it absent (the amnesia
claim)?

  level probe (control):       h[t] -> topmass, entropy.   Expected to SUCCEED.
  derivative probe (the test): h[t] -> d_conf, dd_conf.    THE test.
  routing probe:               h[t] -> source_trust.       Secondary.

A LINEAR probe is the right tool: if d_conf is only recoverable non-linearly,
the info is present-but-encoded — still "amnesic" for the purposes of cheap
routing. Linear readability is what a selector head would cheaply exploit.
"""
from __future__ import annotations
from dataclasses import dataclass
import warnings
import torch
import torch.nn.functional as F


@dataclass
class ProbeResult:
    target_name: str
    mse: float
    r2: float
    n_train: int
    n_test: int


@dataclass(frozen=True, slots=True)
class ProbeSplitError(ValueError):
    n_train: int
    n_test: int

    def __str__(self) -> str:
        return (
            "probe split must contain at least one train and one test point "
            f"(got n_train={self.n_train}, n_test={self.n_test})"
        )


def _flat_probe(
    hid: torch.Tensor,      # [P, d]
    tgt: torch.Tensor,      # [P,] or [P, k]
    is_train: torch.Tensor, # [P] bool
    wd: float = 1e-4,
) -> tuple[float, float]:
    """Fit linear hid->tgt on is_train positions, eval MSE & R^2 on the rest.
    Targets standardized using TRAIN stats; metrics reported in original units."""
    hid = hid.detach().cpu().to(torch.float64)
    tgt = tgt.detach().cpu().to(torch.float64)
    is_train = is_train.detach().cpu().to(dtype=torch.bool)
    n_train = int(is_train.sum().item())
    n_test = int((~is_train).sum().item())
    if n_train == 0 or n_test == 0:
        raise ProbeSplitError(n_train=n_train, n_test=n_test)
    if tgt.dim() == 1:
        tgt = tgt.unsqueeze(-1)
    y_mean = tgt[is_train].mean(0, keepdim=True)
    y_std = tgt[is_train].std(0, keepdim=True).clamp_min(1e-6)
    Yn = (tgt - y_mean) / y_std
    x = torch.cat(
        [hid, torch.ones((hid.shape[0], 1), dtype=hid.dtype, device=hid.device)],
        dim=1,
    )
    x_train = x[is_train]
    y_train = Yn[is_train]
    eye = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
    weights = torch.linalg.solve(
        x_train.T @ x_train + wd * eye,
        x_train.T @ y_train,
    )
    with torch.no_grad():
        pred = (x[~is_train] @ weights) * y_std + y_mean
        truth = tgt[~is_train]
        mse = F.mse_loss(pred.to(torch.float32), truth.to(torch.float32)).item()
        ss_res = ((truth - pred) ** 2).sum()
        ss_tot = ((truth - truth.mean()) ** 2).sum()
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot.item() > 0 else 0.0
    return mse, r2


def _warn_if_legacy_probe_args_used(
    n_epochs: int | None,
    lr: float | None,
    device: str | None,
) -> None:
    legacy_args = []
    if n_epochs is not None:
        legacy_args.append("n_epochs")
    if lr is not None:
        legacy_args.append("lr")
    if device is not None:
        legacy_args.append("device")
    if not legacy_args:
        return
    names = ", ".join(legacy_args)
    warnings.warn(
        f"linear probe now uses closed-form CPU ridge; ignored compatibility args: {names}",
        DeprecationWarning,
        stacklevel=3,
    )


def linear_probe(
    hidden: torch.Tensor, target: torch.Tensor, target_name: str,
    train_mask: torch.Tensor,
    n_epochs: int | None = None, lr: float | None = None,
    device: str | None = None, *, weight_decay: float = 1e-4,
) -> ProbeResult:
    _warn_if_legacy_probe_args_used(n_epochs=n_epochs, lr=lr, device=device)
    hid = hidden.reshape(-1, hidden.shape[-1])
    if target.dim() == 2:
        tgt = target.reshape(-1)
    else:
        tgt = target.reshape(-1, target.shape[-1])
    is_train = train_mask.reshape(-1)
    mse, r2 = _flat_probe(hid, tgt, is_train, weight_decay)
    return ProbeResult(
        target_name,
        mse,
        r2,
        n_train=int(is_train.sum()),
        n_test=int((~is_train).sum()),
    )


# --------------------------------------------------------------------------
# Experiment driver: the full gating procedure from SPEC-000.
# --------------------------------------------------------------------------

@dataclass
class ProbeExperimentResult:
    level: dict       # target_name -> ProbeResult  (controls; should succeed)
    derivative: dict  # target_name -> ProbeResult  (THE test)
    routing: dict
    n_train_streams: int
    n_test_streams: int
    backbone_params: int
    backbone_seed: int
    probe_train_seed_base: int
    probe_test_seed_base: int


def run_probe_experiment(
    backbone,                # a trained, frozen DenseTransformer
    stream,                  # BayesianEvidenceStream
    n_train_streams: int = 200,
    n_test_streams: int = 80,
    seed: int = 0,
    device: str = "cpu",
    probe_train_seed_base: int = 10_000_000,
    probe_test_seed_base: int = 20_000_000,
) -> ProbeExperimentResult:
    """Collect hidden states over held-out streams; probe each target.

    SPLIT: probe fits on train streams, evaluates on DISJOINT test streams
    (both restricted to real-evidence positions). The probe must generalize
    to streams the backbone was not probe-fit on.

    Targets:
      level:       topmass, entropy         (control)
      derivative:  d_conf, dd_conf          (THE test)
      routing:     source_trust per source
    """
    backbone = backbone.to(device).eval()
    params = sum(p.numel() for p in backbone.parameters())

    def collect(n_streams, base_seed):
        Hs, topmass, entropy, d_conf, dd_conf, st, real = ([] for _ in range(7))
        with torch.no_grad():
            for i in range(n_streams):
                s = stream.generate(seed=base_seed + i)
                tok = torch.as_tensor(s.tokens).long().unsqueeze(0).to(device)
                h = backbone(tok).hidden[0].cpu()  # [T,d]
                Hs.append(h)
                topmass.append(torch.as_tensor(s.posterior.max(1), dtype=torch.float32))
                entropy.append(torch.as_tensor(s.entropy, dtype=torch.float32))
                d_conf.append(torch.as_tensor(s.d_conf, dtype=torch.float32))
                dd_conf.append(torch.as_tensor(s.dd_conf, dtype=torch.float32))
                st.append(torch.as_tensor(s.source_trust, dtype=torch.float32))  # [T,S]
                m = torch.zeros(stream.T, dtype=torch.bool)
                m[:s.metadata["natural_len"]] = True
                real.append(m)
        return (torch.stack(Hs), torch.stack(topmass), torch.stack(entropy),
                torch.stack(d_conf), torch.stack(dd_conf), torch.stack(st),
                torch.stack(real))

    Htr, tm_tr, en_tr, dc_tr, ddc_tr, st_tr, real_tr = collect(
        n_train_streams,
        probe_train_seed_base,
    )
    Hte, tm_te, en_te, dc_te, ddc_te, st_te, real_te = collect(
        n_test_streams,
        probe_test_seed_base,
    )

    H = torch.cat([Htr, Hte])                                   # [N,T,d]
    real = torch.cat([real_tr, real_te])                        # [N,T]
    train_stream = torch.zeros(H.shape[0], dtype=torch.bool)
    train_stream[:n_train_streams] = True
    is_train_pos = train_stream.unsqueeze(1) & real             # [N,T]
    test_pos = (~train_stream).unsqueeze(1) & real
    keep = is_train_pos | test_pos                              # real positions only

    flat_train = is_train_pos[keep]                             # [P] bool

    def run(name, target_2d):
        tgt = target_2d[keep]                                   # [P,]
        hid = H[keep]                                           # [P,d]
        mse, r2 = _flat_probe(hid, tgt, flat_train)
        return ProbeResult(name, mse, r2,
                           n_train=int(flat_train.sum()),
                           n_test=int((~flat_train).sum()))

    level = {"topmass": run("topmass", torch.cat([tm_tr, tm_te])),
             "entropy": run("entropy", torch.cat([en_tr, en_te]))}
    deriv = {"d_conf": run("d_conf", torch.cat([dc_tr, dc_te])),
             "dd_conf": run("dd_conf", torch.cat([ddc_tr, ddc_te]))}
    st_full = torch.cat([st_tr, st_te])                         # [N,T,S]
    routing = {f"source_trust_{s}": run(f"source_trust_{s}", st_full[..., s])
               for s in range(stream.S)}

    return ProbeExperimentResult(level=level, derivative=deriv, routing=routing,
                                 n_train_streams=n_train_streams,
                                 n_test_streams=n_test_streams,
                                 backbone_params=params, backbone_seed=seed,
                                 probe_train_seed_base=probe_train_seed_base,
                                 probe_test_seed_base=probe_test_seed_base)
