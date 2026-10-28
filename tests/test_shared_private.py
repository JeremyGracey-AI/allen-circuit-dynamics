import numpy as np

from allen_circuit_dynamics.models.shared_private import SharedPrivateFA
from allen_circuit_dynamics.preprocess import (
    PreprocessConfig,
    apply_scaler,
    fit_scaler,
    transform_counts,
)


def _prep(ds):
    x1, _ = transform_counts(ds.x1, PreprocessConfig())
    x2, _ = transform_counts(ds.x2, PreprocessConfig())
    return apply_scaler(x1, fit_scaler(x1)), apply_scaler(x2, fit_scaler(x2))


def test_mask_is_enforced(small_synth):
    ds, _ = small_synth
    a, b = _prep(ds)
    m = SharedPrivateFA(d_shared=3, d_private=(2, 2), max_iter=50).fit(a, b)
    n1 = a.shape[1]
    assert np.all(m.loadings_[:n1, 5:] == 0)  # pop1 rows never load on private2
    assert np.all(m.loadings_[n1:, 3:5] == 0)  # pop2 rows never load on private1


def test_loglik_increases_monotonically(small_synth):
    ds, _ = small_synth
    a, b = _prep(ds)
    m = SharedPrivateFA(d_shared=3, d_private=(2, 2), max_iter=100).fit(a, b)
    diffs = np.diff(m.loglik_trace_)
    assert np.all(diffs > -1e-6), "EM must not decrease the likelihood"


def test_heldout_prefers_true_shared_dim(small_synth):
    """Held-out log-likelihood should peak at the planted d_shared=3, not at 0 or 6."""
    ds, _ = small_synth
    a, b = _prep(ds)
    n = a.shape[0]
    tr, te = slice(0, int(0.7 * n)), slice(int(0.7 * n), n)
    scores = {}
    for d in (0, 1, 3, 6):
        m = SharedPrivateFA(d_shared=d, d_private=(2, 2)).fit(a[tr], b[tr])
        scores[d] = m.score(a[te], b[te])
    assert scores[3] > scores[0]
    assert scores[3] > scores[1]
    assert scores[3] >= scores[6] - 0.05  # not meaningfully worse than overfitting


def test_variance_partition_sums_to_one(small_synth):
    ds, _ = small_synth
    a, b = _prep(ds)
    m = SharedPrivateFA(d_shared=3, d_private=(2, 2)).fit(a, b)
    vp = m.variance_partition()
    for pop in ("pop1", "pop2"):
        assert abs(vp[pop]["shared"] + vp[pop]["private"] + vp[pop]["noise"] - 1.0) < 1e-9
        assert vp[pop]["shared"] > vp[pop]["private"]  # planted shared_gain > private_gain


def test_shared_latents_track_planted_shared(small_synth):
    """Canonical correlation between recovered and planted shared latents is high."""
    ds, truth = small_synth
    a, b = _prep(ds)
    m = SharedPrivateFA(d_shared=3, d_private=(2, 2)).fit(a, b)
    z_hat = m.transform(a, b)["shared"]
    z_true = truth["z_shared"]
    # subspace alignment via least-squares R^2 of true latents on recovered ones
    beta, *_ = np.linalg.lstsq(z_hat, z_true, rcond=None)
    resid = z_true - z_hat @ beta
    r2 = 1 - resid.var(axis=0) / z_true.var(axis=0)
    assert r2.mean() > 0.55 and np.all(r2 > 0.45), r2


def test_cross_prediction_beats_mean(small_synth):
    ds, _ = small_synth
    a, b = _prep(ds)
    n = a.shape[0]
    tr, te = slice(0, int(0.7 * n)), slice(int(0.7 * n), n)
    m = SharedPrivateFA(d_shared=3, d_private=(2, 2)).fit(a[tr], b[tr])
    pred = m.predict_cross(a[te])
    r2 = 1 - np.sum((b[te] - pred) ** 2) / np.sum((b[te] - b[te].mean(0)) ** 2)
    assert r2 > 0.02
