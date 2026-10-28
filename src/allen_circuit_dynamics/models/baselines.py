"""Published-style baselines the shared/private model is benchmarked against.

* :func:`reduced_rank_regression` — the "communication subspace" estimator
  (Semedo et al., 2019): predict population 2 from population 1 with a
  rank-constrained linear map. Its held-out R² as a function of rank is the
  standard yardstick for *how many shared dimensions* two populations use.
* :func:`cca_shared` — classical CCA (via sklearn) for the number of
  significantly correlated canonical pairs.
* :func:`per_population_fa` — plain factor analysis on each population alone,
  which absorbs shared and private variance together; the gap between its
  held-out likelihood and the shared/private model's is the evidence that the
  partition is worth having.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import FactorAnalysis


def ridge_solve(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Return B minimising ||y - x B||² + alpha ||B||², shapes (p, q)."""
    p = x.shape[1]
    return linalg.solve(x.T @ x + alpha * np.eye(p), x.T @ y, assume_a="pos")


def reduced_rank_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    ranks: list[int],
    alpha: float = 1.0,
) -> dict[int, float]:
    """Held-out R² of predicting ``y`` from ``x`` at each rank.

    Fits the full ridge solution once, then projects it onto the top-``r``
    principal directions of the fitted values (Izenman's RRR). Training fit is
    monotone in rank; the held-out curve typically rises and then flattens or
    dips, and the rank where it flattens is the estimate of the number of
    shared dimensions.
    """
    xm, ym = x_train.mean(0), y_train.mean(0)
    xtr, ytr = x_train - xm, y_train - ym
    xte, yte = x_test - xm, y_test - ym
    b_full = ridge_solve(xtr, ytr, alpha)
    yhat_tr = xtr @ b_full
    _, _, vt = linalg.svd(yhat_tr, full_matrices=False)
    ss_tot = float(np.sum(yte**2))
    out = {}
    for r in ranks:
        v = vt[:r].T  # (q, r)
        b_r = b_full @ v @ v.T
        resid = yte - xte @ b_r
        out[int(r)] = 1.0 - float(np.sum(resid**2)) / ss_tot
    return out


def cca_shared(
    x1_train: np.ndarray,
    x2_train: np.ndarray,
    x1_test: np.ndarray,
    x2_test: np.ndarray,
    n_components: int,
) -> dict:
    """Held-out canonical correlations for the first ``n_components`` pairs."""
    k = min(n_components, x1_train.shape[1], x2_train.shape[1])
    cca = CCA(n_components=k, max_iter=1000).fit(x1_train, x2_train)
    a, b = cca.transform(x1_test, x2_test)
    corrs = [float(np.corrcoef(a[:, i], b[:, i])[0, 1]) for i in range(k)]
    return {"n_components": k, "heldout_canonical_corr": corrs}


def per_population_fa(x_train: np.ndarray, x_test: np.ndarray, n_components: int) -> dict:
    """Plain FA on one population; mean held-out log-likelihood per bin."""
    k = min(n_components, x_train.shape[1])
    fa = FactorAnalysis(n_components=k).fit(x_train)
    return {"n_components": k, "heldout_loglik_per_bin": float(fa.score(x_test))}
