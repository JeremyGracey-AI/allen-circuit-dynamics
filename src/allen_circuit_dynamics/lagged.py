"""Lag alignment that respects cross-validation blocks.

To ask whether population 2 follows population 1 by ``lag`` bins, pair
``x1[t]`` with ``x2[t + lag]`` (positive lag: pop 2 lags pop 1). A pair is kept
only when both bins fall inside the same block, so a lag never reaches across
a gap between stimulus presentations or across a CV boundary.

Everything here is pure array bookkeeping; the models never see a lag.
"""

from __future__ import annotations

import numpy as np


def lag_pairs(blocks: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices ``(i, j)`` with ``j = i + lag`` and ``blocks[i] == blocks[j]``."""
    n = blocks.shape[0]
    if abs(lag) >= n:
        raise ValueError(f"lag {lag} exceeds the series length {n}.")
    if lag >= 0:
        i = np.arange(0, n - lag)
    else:
        i = np.arange(-lag, n)
    j = i + lag
    keep = blocks[i] == blocks[j]
    return i[keep], j[keep]


def align(
    x1: np.ndarray,
    x2: np.ndarray,
    blocks: np.ndarray,
    lag: int,
    *extra: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Return ``x1[i]``, ``x2[j]``, ``blocks[i]`` and each ``extra[i]`` for the kept pairs.

    ``extra`` arrays (e.g. behavior) are indexed on population 1's clock.
    """
    i, j = lag_pairs(blocks, lag)
    out = [x1[i], x2[j], blocks[i]]
    out.extend(e[i] for e in extra)
    return tuple(out)


def circular_shift_within_blocks(
    x: np.ndarray, blocks: np.ndarray, rng: np.random.Generator, min_shift: int = 1
) -> np.ndarray:
    """Roll each block of ``x`` by its own random offset (>= ``min_shift`` bins).

    Preserves every within-block autocorrelation while destroying the bin-by-bin
    alignment with any other series: the null for a lag profile.
    """
    out = np.empty_like(x)
    for b in np.unique(blocks):
        idx = np.flatnonzero(blocks == b)
        n = idx.size
        if n <= 2 * min_shift:
            out[idx] = x[idx]
            continue
        k = int(rng.integers(min_shift, n - min_shift))
        out[idx] = np.roll(x[idx], k, axis=0)
    return out
