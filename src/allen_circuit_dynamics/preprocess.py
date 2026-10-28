"""Turn binned spike counts into something a Gaussian latent model can eat.

Steps, each one recorded in the returned provenance list:

1. drop units whose mean rate is below ``min_rate_hz`` (near-silent units add
   noise dimensions and nothing else),
2. square-root transform counts (variance-stabilising for Poisson-ish data),
3. optionally smooth with a causal boxcar of ``smooth_bins`` bins,
4. z-score each unit using **training-fold** statistics only when called from
   the cross-validation loop (see :func:`fit_scaler` / :func:`apply_scaler`).

Nothing is dropped silently: the unit filter returns the kept indices and the
count of dropped units.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PreprocessConfig:
    min_rate_hz: float = 0.5
    sqrt_transform: bool = True
    smooth_bins: int = 1


@dataclass(frozen=True)
class Scaler:
    mean: np.ndarray
    std: np.ndarray


def filter_units(x: np.ndarray, bin_s: float, min_rate_hz: float) -> tuple[np.ndarray, dict]:
    rates = x.mean(axis=0) / bin_s
    keep = rates >= min_rate_hz
    info = {
        "n_in": int(x.shape[1]),
        "n_kept": int(keep.sum()),
        "n_dropped_low_rate": int((~keep).sum()),
        "min_rate_hz": min_rate_hz,
    }
    if keep.sum() < 2:
        raise ValueError(f"Only {keep.sum()} units pass the rate filter; lower min_rate_hz.")
    return x[:, keep], info


def transform_counts(x: np.ndarray, cfg: PreprocessConfig) -> tuple[np.ndarray, list[str]]:
    steps = []
    out = x.astype(float)
    if cfg.sqrt_transform:
        out = np.sqrt(out)
        steps.append("sqrt")
    if cfg.smooth_bins > 1:
        kernel = np.ones(cfg.smooth_bins) / cfg.smooth_bins
        out = np.apply_along_axis(lambda c: np.convolve(c, kernel, mode="full")[: c.size], 0, out)
        steps.append(f"causal_boxcar[{cfg.smooth_bins}]")
    return out, steps


def fit_scaler(x_train: np.ndarray) -> Scaler:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return Scaler(mean=mean, std=std)


def apply_scaler(x: np.ndarray, sc: Scaler) -> np.ndarray:
    return (x - sc.mean) / sc.std
