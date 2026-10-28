"""Which subspace carries the behavior?

Ridge-regress the behavioral signal on each latent set (shared, private-1,
private-2, shared+private, raw populations) with **block-grouped**
cross-validation, and compare against a circular-shift null where the
behavioral trace is rotated by a random offset within each test fold. The
null preserves the autocorrelation of behavior, which a plain permutation
would destroy and which is what makes naive decoders look good.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold


@dataclass(frozen=True)
class DecodeResult:
    r2_mean: float
    r2_std: float
    r2_folds: list[float]
    null_r2_mean: float
    null_r2_p95: float
    n_features: int


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def decode_cv(
    features: np.ndarray,
    behavior: np.ndarray,
    blocks: np.ndarray,
    n_splits: int = 5,
    alpha: float = 1.0,
    n_null: int = 50,
    seed: int = 0,
) -> DecodeResult:
    rng = np.random.default_rng(seed)
    gkf = GroupKFold(n_splits=min(n_splits, np.unique(blocks).size))
    r2s, nulls = [], []
    for tr, te in gkf.split(features, behavior, groups=blocks):
        model = Ridge(alpha=alpha).fit(features[tr], behavior[tr])
        yhat = model.predict(features[te])
        r2s.append(_r2(behavior[te], yhat))
        n_te = te.size
        for _ in range(n_null):
            shift = int(rng.integers(n_te // 10, n_te - n_te // 10))
            nulls.append(_r2(np.roll(behavior[te], shift), yhat))
    nulls_arr = np.asarray(nulls)
    return DecodeResult(
        r2_mean=float(np.mean(r2s)),
        r2_std=float(np.std(r2s)),
        r2_folds=[float(v) for v in r2s],
        null_r2_mean=float(np.mean(nulls_arr)),
        null_r2_p95=float(np.percentile(nulls_arr, 95)),
        n_features=int(features.shape[1]),
    )


def decode_all(
    latents: dict[str, np.ndarray],
    x1: np.ndarray,
    x2: np.ndarray,
    behavior: np.ndarray,
    blocks: np.ndarray,
    **kw,
) -> dict[str, DecodeResult]:
    """Decode from every subspace of interest; the raw populations are the ceiling."""
    sets = {
        "shared": latents["shared"],
        "private1": latents["private1"],
        "private2": latents["private2"],
        "shared+private1": np.hstack([latents["shared"], latents["private1"]]),
        "all_latents": np.hstack([latents["shared"], latents["private1"], latents["private2"]]),
        "raw_pop1": x1,
        "raw_pop2": x2,
        "raw_both": np.hstack([x1, x2]),
    }
    return {k: decode_cv(v, behavior, blocks, **kw) for k, v in sets.items() if v.shape[1] > 0}
