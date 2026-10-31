"""Connectivity-constrained linear dynamics across areas (Experiment 2).

Given per-area population latents ``s_a(t)`` (one block per area, stacked
into ``S`` of shape ``(n_bins, K)``), fit a one-step linear model::

    S[t+1] = S[t] @ A + b + noise

where ``A`` is partitioned into area-by-area coupling blocks. The anatomical
prior enters as a penalty on each block proportional to how *weak* the
projection between those areas is in the connectivity atlas::

    penalty(A) = lambda * c * sum_{a,b} (1 - conn[a, b]) * ||A_block(a,b)||_F^2

so blocks between areas with no projection are pushed toward zero and blocks
between strongly connected areas are left nearly free. ``c = trace(X'X) / K``
(the mean diagonal of the data Gram matrix) makes ``lambda`` dimensionless:
``lambda = 1`` means the penalty on a fully unconnected block equals the
typical data curvature, whatever the number of bins or the latent scale.
Without ``c`` the same ``lambda`` is ten times weaker on a session with ten
times more bins, which is exactly what happened on the first real run. ``conn`` is a
row-normalised projection-density matrix in ``[0, 1]`` (diagonal = 1, since
within-area recurrence is always allowed).

The comparison that matters is held-out one-step prediction (per-area R²) of
the constrained model vs. the unconstrained ridge model vs. a model fit with a
*shuffled* connectivity matrix. If the atlas carries information about the
dynamics, the true-atlas model should beat the shuffled-atlas model at the
same penalty strength. If it doesn't, that is a finding too.

This module works on any latents you hand it; building them from the
Neuropixels areas is the loader's job.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg


@dataclass
class CoupledLDS:
    area_dims: list[int]
    conn: np.ndarray | None = None
    lam: float = 0.1  # dimensionless; see module docstring
    ridge: float = 1e-3  # dimensionless baseline ridge on every entry of A

    A_: np.ndarray | None = None
    b_: np.ndarray | None = None

    @property
    def k(self) -> int:
        return int(sum(self.area_dims))

    def _area_of_column(self) -> np.ndarray:
        return np.repeat(np.arange(len(self.area_dims)), self.area_dims)

    def _penalty_matrix(self) -> np.ndarray:
        """Per-entry penalty weights for A, shape (k, k)."""
        area = self._area_of_column()
        n_areas = len(self.area_dims)
        if self.conn is None:
            conn = np.ones((n_areas, n_areas))
        else:
            conn = np.asarray(self.conn, dtype=float)
            if conn.shape != (n_areas, n_areas):
                raise ValueError("conn must be (n_areas, n_areas).")
            if np.any(conn < 0) or np.any(conn > 1):
                raise ValueError("conn entries must lie in [0, 1].")
        w = 1.0 - conn[area][:, area]  # (k, k): penalty on A[i, j] from area(i) -> area(j)
        return self.ridge + self.lam * w

    def fit(self, s: np.ndarray, blocks: np.ndarray) -> CoupledLDS:
        """Fit on consecutive pairs that lie within the same block."""
        if s.shape[1] != self.k:
            raise ValueError(f"S has {s.shape[1]} columns, area_dims sum to {self.k}.")
        x, y = _pairs(s, blocks)
        self.b_ = y.mean(0) - x.mean(0)  # intercept absorbs the drift
        xc, yc = x - x.mean(0), y - y.mean(0)
        xtx = xc.T @ xc
        xty = xc.T @ yc
        # Scale-free penalty: express ridge and lambda relative to the mean data curvature.
        scale = float(np.trace(xtx)) / self.k
        pen = self._penalty_matrix() * scale
        # Column j of A has its own penalty diag; solve each column separately.
        A = np.zeros((self.k, self.k))
        for j in range(self.k):
            A[:, j] = linalg.solve(xtx + np.diag(pen[:, j]), xty[:, j], assume_a="pos")
        self.A_ = A
        return self

    def score(self, s: np.ndarray, blocks: np.ndarray) -> dict:
        """Held-out one-step R² overall and per area."""
        x, y = _pairs(s, blocks)
        yhat = x @ self.A_ + self.b_
        resid = y - yhat
        ss_res = np.sum(resid**2, axis=0)
        ss_tot = np.sum((y - y.mean(0)) ** 2, axis=0)
        r2_col = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
        area = self._area_of_column()
        per_area = [float(np.mean(r2_col[area == a])) for a in range(len(self.area_dims))]
        return {"r2_overall": float(1.0 - ss_res.sum() / ss_tot.sum()), "r2_per_area": per_area}

    def block_norms(self) -> np.ndarray:
        """Frobenius norm of each area->area coupling block, shape (n_areas, n_areas)."""
        area = self._area_of_column()
        n = len(self.area_dims)
        out = np.zeros((n, n))
        for a in range(n):
            for b in range(n):
                out[a, b] = np.linalg.norm(self.A_[np.ix_(area == a, area == b)])
        return out


def _pairs(s: np.ndarray, blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    same = blocks[1:] == blocks[:-1]
    return s[:-1][same], s[1:][same]


def shuffle_conn(conn: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute the off-diagonal structure while keeping the diagonal at 1."""
    n = conn.shape[0]
    off = conn[~np.eye(n, dtype=bool)]
    out = np.eye(n)
    out[~np.eye(n, dtype=bool)] = rng.permutation(off)
    return out
