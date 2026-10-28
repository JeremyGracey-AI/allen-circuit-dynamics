"""Shared/private factor analysis for two simultaneously recorded populations.

Generative model (all latents standard normal, noise diagonal)::

    x1 = W1 z + V1 u1 + mu1 + e1
    x2 = W2 z + V2 u2 + mu2 + e2

``z`` (``d_shared`` dims) drives both populations; ``u1``/``u2`` (``d_private``
dims each) drive only their own. This is factor analysis on the concatenated
vector ``[x1, x2]`` with a block-sparse loading matrix::

    L = [[W1, V1, 0 ],
         [W2, 0,  V2]]

and is fit by EM with the zero pattern enforced in every M-step. It is the
static special case of models such as probabilistic CCA (Bach & Jordan, 2005)
with explicit private subspaces, and of DLAG (Gokcen et al., 2022) without the
time-delay structure.

What it gives you:

* a variance partition per population — shared / private / noise fractions,
* posterior latents (shared, private-1, private-2) for downstream decoding,
* a held-out log-likelihood for choosing ``d_shared`` and ``d_private``,
* a cross-population prediction (E[x2 | x1]) that only the shared block can
  carry, so it benchmarks directly against reduced-rank regression.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import linalg


@dataclass
class SharedPrivateFA:
    d_shared: int = 2
    d_private: tuple[int, int] = (1, 1)
    max_iter: int = 500
    tol: float = 1e-5
    min_noise: float = 1e-4
    seed: int = 0

    # fitted
    n_units: tuple[int, int] = field(default=(0, 0), init=False)
    mean_: np.ndarray = field(default=None, init=False, repr=False)
    loadings_: np.ndarray = field(default=None, init=False, repr=False)
    noise_: np.ndarray = field(default=None, init=False, repr=False)
    mask_: np.ndarray = field(default=None, init=False, repr=False)
    loglik_trace_: list[float] = field(default_factory=list, init=False, repr=False)
    n_iter_: int = field(default=0, init=False)

    # ------------------------------------------------------------------ setup
    @property
    def k(self) -> int:
        return self.d_shared + sum(self.d_private)

    def _latent_slices(self) -> dict[str, slice]:
        s, (p1, p2) = self.d_shared, self.d_private
        return {
            "shared": slice(0, s),
            "private1": slice(s, s + p1),
            "private2": slice(s + p1, s + p1 + p2),
        }

    def _build_mask(self, n1: int, n2: int) -> np.ndarray:
        s, (p1, _) = self.d_shared, self.d_private
        mask = np.zeros((n1 + n2, self.k), dtype=bool)
        mask[:n1, :s] = True
        mask[n1:, :s] = True
        mask[:n1, s : s + p1] = True
        mask[n1:, s + p1 :] = True
        return mask

    def _init(self, x1c: np.ndarray, x2c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Shared loadings from the cross-covariance SVD, private from residual PCA."""
        n = x1c.shape[0]
        n1, n2 = x1c.shape[1], x2c.shape[1]
        s, (p1, p2) = self.d_shared, self.d_private
        rng = np.random.default_rng(self.seed)
        L = np.zeros((n1 + n2, self.k))

        if s > 0:
            c12 = x1c.T @ x2c / n
            u, sv, vt = linalg.svd(c12, full_matrices=False)
            r = min(s, sv.size)
            L[:n1, :r] = u[:, :r] * np.sqrt(sv[:r])
            L[n1:, :r] = vt[:r].T * np.sqrt(sv[:r])
            if r < s:  # more shared dims than the cross-cov rank; pad with noise
                L[:, r:s] = 0.1 * rng.standard_normal((n1 + n2, s - r))

        for xc, lo, hi, p in ((x1c, 0, n1, p1), (x2c, n1, n1 + n2, p2)):
            if p == 0:
                continue
            resid = xc - xc @ np.linalg.pinv(L[lo:hi, :s].T) @ L[lo:hi, :s].T if s else xc
            cov = resid.T @ resid / n
            evals, evecs = linalg.eigh(cov)
            idx = np.argsort(evals)[::-1][:p]
            col = s if lo == 0 else s + p1
            L[lo:hi, col : col + p] = evecs[:, idx] * np.sqrt(np.clip(evals[idx], 1e-6, None))

        x = np.hstack([x1c, x2c])
        psi = np.clip(np.var(x, axis=0) - np.sum(L**2, axis=1), self.min_noise, None)
        return L, psi

    # -------------------------------------------------------------------- fit
    def fit(self, x1: np.ndarray, x2: np.ndarray) -> SharedPrivateFA:
        n1, n2 = x1.shape[1], x2.shape[1]
        self.n_units = (n1, n2)
        x = np.hstack([x1, x2]).astype(float)
        self.mean_ = x.mean(axis=0)
        xc = x - self.mean_
        n, d = xc.shape
        self.mask_ = self._build_mask(n1, n2)
        L, psi = self._init(xc[:, :n1], xc[:, n1:])
        k = self.k
        xtx_diag = np.sum(xc**2, axis=0)

        prev = -np.inf
        self.loglik_trace_ = []
        for it in range(self.max_iter):
            # E-step
            sigma_post, m = self._posterior(xc, L, psi)  # (k,k), (n,k)
            e_zz = n * sigma_post + m.T @ m  # sum_t E[z z^T]
            xtm = xc.T @ m  # (d, k)

            # M-step with the block-sparse mask
            L_new = np.zeros_like(L)
            for i in range(d):
                cols = self.mask_[i]
                a = e_zz[np.ix_(cols, cols)]
                b = xtm[i, cols]
                L_new[i, cols] = linalg.solve(a, b, assume_a="pos")
            quad = np.einsum("ik,kl,il->i", L_new, e_zz, L_new)
            psi_new = (xtx_diag - 2 * np.sum(L_new * xtm, axis=1) + quad) / n
            psi_new = np.clip(psi_new, self.min_noise, None)

            L, psi = L_new, psi_new
            ll = self._loglik(xc, L, psi)
            self.loglik_trace_.append(ll)
            self.n_iter_ = it + 1
            if ll - prev < self.tol * max(1.0, abs(ll)) and it > 5:
                break
            prev = ll

        self.loadings_, self.noise_ = L, psi
        assert np.all(L[~self.mask_] == 0.0), "mask violated"
        assert L.shape == (d, k)
        return self

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _posterior(xc: np.ndarray, L: np.ndarray, psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior covariance (shared across bins) and means of the latents."""
        k = L.shape[1]
        lt_psi_inv = L.T / psi  # (k, d)
        sigma_post = linalg.inv(np.eye(k) + lt_psi_inv @ L)
        m = xc @ (lt_psi_inv.T @ sigma_post)  # (n, k)
        return sigma_post, m

    @staticmethod
    def _loglik(xc: np.ndarray, L: np.ndarray, psi: np.ndarray) -> float:
        """Mean Gaussian log-likelihood per bin, via the matrix determinant lemma."""
        n, d = xc.shape
        k = L.shape[1]
        lt_psi_inv = L.T / psi
        a = np.eye(k) + lt_psi_inv @ L
        sign, logdet_a = np.linalg.slogdet(a)
        logdet_c = logdet_a + np.sum(np.log(psi))
        # x^T C^{-1} x via Woodbury: C^{-1} = Psi^{-1} - Psi^{-1} L A^{-1} L^T Psi^{-1}
        y = xc / np.sqrt(psi)
        t = xc @ lt_psi_inv.T  # (n, k)
        quad = np.sum(y**2, axis=1) - np.einsum("ik,kl,il->i", t, linalg.inv(a), t)
        return float(np.mean(-0.5 * (d * np.log(2 * np.pi) + logdet_c + quad)))

    # ---------------------------------------------------------------- outputs
    def score(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Mean held-out log-likelihood per bin."""
        xc = np.hstack([x1, x2]) - self.mean_
        return self._loglik(xc, self.loadings_, self.noise_)

    def transform(self, x1: np.ndarray, x2: np.ndarray) -> dict[str, np.ndarray]:
        """Posterior-mean latents keyed by ``shared`` / ``private1`` / ``private2``."""
        xc = np.hstack([x1, x2]) - self.mean_
        _, m = self._posterior(xc, self.loadings_, self.noise_)
        return {name: m[:, sl] for name, sl in self._latent_slices().items()}

    def variance_partition(self) -> dict[str, dict[str, float]]:
        """Shared / private / noise variance fraction per population."""
        n1, _ = self.n_units
        sl = self._latent_slices()
        out = {}
        for name, rows, priv in (
            ("pop1", slice(0, n1), "private1"),
            ("pop2", slice(n1, None), "private2"),
        ):
            L = self.loadings_[rows]
            shared = float(np.sum(L[:, sl["shared"]] ** 2))
            private = float(np.sum(L[:, sl[priv]] ** 2))
            noise = float(np.sum(self.noise_[rows]))
            total = shared + private + noise
            out[name] = {
                "shared": shared / total,
                "private": private / total,
                "noise": noise / total,
                "total_variance": total,
            }
        return out

    def predict_cross(self, x1: np.ndarray) -> np.ndarray:
        """E[x2 | x1] under the fitted Gaussian; only the shared block carries it."""
        n1, _ = self.n_units
        L, psi = self.loadings_, self.noise_
        c = L @ L.T + np.diag(psi)
        c11, c21 = c[:n1, :n1], c[n1:, :n1]
        x1c = x1 - self.mean_[:n1]
        return self.mean_[n1:] + x1c @ linalg.solve(c11, c21.T, assume_a="pos")
