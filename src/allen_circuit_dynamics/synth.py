"""Synthetic two-population data with a planted shared/private structure.

The generator is the ground truth for the recovery tests. It plants:

* ``d_shared`` latent factors that drive both populations,
* ``d_private`` latent factors per population that drive only that one,
* a behavioral signal that is a linear readout of the *shared* factors plus a
  readout of population 1's *private* factors, so the decoder test has a
  known answer about which subspace carries behavior,
* Poisson spike counts, so the data look like binned Neuropixels units.

Latents are smooth AR(1) processes rather than white noise so block-wise
cross-validation matters, as it does on real recordings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dataset import PopulationDataset


@dataclass(frozen=True)
class SynthConfig:
    n_bins: int = 6000
    n_units: tuple[int, int] = (40, 30)
    d_shared: int = 3
    d_private: tuple[int, int] = (2, 2)
    bin_s: float = 0.05
    n_blocks: int = 12
    ar_coef: float = 0.9
    shared_gain: float = 1.4
    private_gain: float = 1.0
    base_rate_hz: float = 15.0
    behavior_private_weight: float = 0.5
    seed: int = 0


def _ar1(rng: np.random.Generator, n: int, d: int, coef: float) -> np.ndarray:
    """Zero-mean, unit-variance AR(1) latents, shape (n, d)."""
    z = np.zeros((n, d))
    noise_sd = np.sqrt(1.0 - coef**2)
    z[0] = rng.standard_normal(d)
    for t in range(1, n):
        z[t] = coef * z[t - 1] + noise_sd * rng.standard_normal(d)
    return z


def make_synthetic(cfg: SynthConfig | None = None) -> tuple[PopulationDataset, dict]:
    """Return a dataset and the ground-truth latents / loadings used to build it."""
    cfg = cfg or SynthConfig()
    rng = np.random.default_rng(cfg.seed)
    n1, n2 = cfg.n_units
    p1, p2 = cfg.d_private

    z_shared = _ar1(rng, cfg.n_bins, cfg.d_shared, cfg.ar_coef)
    u1 = _ar1(rng, cfg.n_bins, p1, cfg.ar_coef)
    u2 = _ar1(rng, cfg.n_bins, p2, cfg.ar_coef)

    w1 = cfg.shared_gain * rng.standard_normal((cfg.d_shared, n1)) / np.sqrt(cfg.d_shared)
    w2 = cfg.shared_gain * rng.standard_normal((cfg.d_shared, n2)) / np.sqrt(cfg.d_shared)
    v1 = cfg.private_gain * rng.standard_normal((p1, n1)) / np.sqrt(max(p1, 1))
    v2 = cfg.private_gain * rng.standard_normal((p2, n2)) / np.sqrt(max(p2, 1))

    log_rate1 = np.log(cfg.base_rate_hz * cfg.bin_s) + 0.5 * (z_shared @ w1 + u1 @ v1)
    log_rate2 = np.log(cfg.base_rate_hz * cfg.bin_s) + 0.5 * (z_shared @ w2 + u2 @ v2)
    x1 = rng.poisson(np.exp(log_rate1)).astype(float)
    x2 = rng.poisson(np.exp(log_rate2)).astype(float)

    b_shared = rng.standard_normal(cfg.d_shared)
    b_priv1 = rng.standard_normal(p1) if p1 else np.zeros(0)
    behavior = z_shared @ b_shared + cfg.behavior_private_weight * (u1 @ b_priv1)
    behavior = behavior + 0.3 * rng.standard_normal(cfg.n_bins)

    blocks = np.floor(np.arange(cfg.n_bins) * cfg.n_blocks / cfg.n_bins).astype(int)

    ds = PopulationDataset(
        x1=x1,
        x2=x2,
        behavior=behavior,
        blocks=blocks,
        bin_s=cfg.bin_s,
        pop_names=("synthA", "synthB"),
        meta={"source": "synthetic", "config": cfg.__dict__},
    )
    truth = {
        "z_shared": z_shared,
        "u1": u1,
        "u2": u2,
        "w1": w1,
        "w2": w2,
        "v1": v1,
        "v2": v2,
        "b_shared": b_shared,
        "b_priv1": b_priv1,
    }
    return ds, truth


@dataclass(frozen=True)
class CoupledSynthConfig:
    """Multi-area latent VAR(1) whose coupling respects a planted connectivity matrix."""

    n_bins: int = 1200
    area_dims: tuple[int, ...] = (8, 8, 8, 8, 8, 8)
    n_blocks: int = 12
    self_coef: float = 0.85
    cross_gain: float = 0.6
    noise_sd: float = 0.5
    seed: int = 0


def make_synthetic_coupled(cfg: CoupledSynthConfig | None = None) -> dict:
    """Return latents ``s`` (n_bins, K), ``blocks``, the planted ``conn`` and true ``A``.

    ``conn`` is a random sparse [0,1] projection-density matrix (diagonal 1). The
    true coupling block from area a to area b is scaled by ``conn[a, b]``, so an
    atlas-informed fit should recover the dynamics better than a shuffled one.
    """
    cfg = cfg or CoupledSynthConfig()
    rng = np.random.default_rng(cfg.seed)
    dims = list(cfg.area_dims)
    n_areas, k = len(dims), sum(dims)
    conn = np.eye(n_areas)
    for a in range(n_areas):
        for b in range(n_areas):
            if a != b:
                conn[a, b] = rng.choice([0.0, 0.0, 0.3, 1.0])
    area = np.repeat(np.arange(n_areas), dims)
    a_true = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if area[i] == area[j]:
                a_true[i, j] = cfg.self_coef if i == j else 0.05 * rng.standard_normal()
            else:
                a_true[i, j] = (
                    cfg.cross_gain * conn[area[i], area[j]] * rng.standard_normal() / np.sqrt(k)
                )
    # keep it stable
    rho = np.max(np.abs(np.linalg.eigvals(a_true)))
    if rho >= 0.98:
        a_true *= 0.95 / rho
    s = np.zeros((cfg.n_bins, k))
    s[0] = rng.standard_normal(k)
    for t in range(1, cfg.n_bins):
        s[t] = s[t - 1] @ a_true + cfg.noise_sd * rng.standard_normal(k)
    blocks = np.floor(np.arange(cfg.n_bins) * cfg.n_blocks / cfg.n_bins).astype(int)
    return {
        "s": s,
        "blocks": blocks,
        "conn": conn,
        "A_true": a_true,
        "area_dims": dims,
        "area_names": [f"area{i}" for i in range(n_areas)],
    }


# ----------------------------------------------------------------------------- Experiment 3


@dataclass(frozen=True)
class ForceSynthConfig:
    """Calcium-like two-population data whose shared factors lead a lever by ``lead_bins``."""

    n_bins: int = 6000
    n_units: tuple[int, int] = (60, 40)  # unequal on purpose: count matching must handle it
    d_shared: int = 3
    d_private: tuple[int, int] = (2, 2)
    ar_coef: float = 0.9
    lead_bins: int = 4  # lever follows the shared factors by this many bins
    noise_sd: float = 1.0
    lever_noise_sd: float = 0.5
    n_blocks: int = 12
    seed: int = 0


def make_synthetic_force(cfg: ForceSynthConfig | None = None) -> tuple[PopulationDataset, dict]:
    cfg = cfg or ForceSynthConfig()
    rng = np.random.default_rng(cfg.seed)
    z = _ar1(rng, cfg.n_bins, cfg.d_shared, cfg.ar_coef)
    u1 = _ar1(rng, cfg.n_bins, cfg.d_private[0], cfg.ar_coef)
    u2 = _ar1(rng, cfg.n_bins, cfg.d_private[1], cfg.ar_coef)
    n1, n2 = cfg.n_units
    W1, W2 = rng.normal(size=(cfg.d_shared, n1)), rng.normal(size=(cfg.d_shared, n2))
    V1, V2 = rng.normal(size=(cfg.d_private[0], n1)), rng.normal(size=(cfg.d_private[1], n2))
    x1 = z @ W1 + u1 @ V1 + cfg.noise_sd * rng.normal(size=(cfg.n_bins, n1))
    x2 = z @ W2 + u2 @ V2 + cfg.noise_sd * rng.normal(size=(cfg.n_bins, n2))
    # lever[t] = readout of z[t - lead]  ->  neural leads the lever by `lead_bins`
    w = rng.normal(size=cfg.d_shared)
    drive = z @ w
    lever = np.roll(drive, cfg.lead_bins)
    lever[: cfg.lead_bins] = drive[0]
    lever = lever + cfg.lever_noise_sd * rng.normal(size=cfg.n_bins)
    blocks = np.minimum(np.arange(cfg.n_bins) * cfg.n_blocks // cfg.n_bins, cfg.n_blocks - 1)
    context = (np.arange(cfg.n_bins) // (cfg.n_bins // 6)) % 2  # alternating blocks, like the task
    motion = 0.3 * lever + rng.normal(size=cfg.n_bins)
    ds = PopulationDataset(
        x1=x1,
        x2=x2,
        behavior=lever,
        blocks=blocks,
        bin_s=0.1,
        pop_names=("D1", "D2"),
        meta={"source": "synthetic_force", "planted": {"lead_bins": cfg.lead_bins}},
        extras={"context": context.astype(float), "motion": motion},
    )
    truth = {"z": z, "u1": u1, "u2": u2, "lead_bins": cfg.lead_bins}
    return ds, truth
