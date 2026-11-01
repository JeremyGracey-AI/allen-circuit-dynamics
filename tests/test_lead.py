import numpy as np
import pytest

from allen_circuit_dynamics.aind_opto import OptoSession
from allen_circuit_dynamics.experiments_lead import (
    LeadConfig,
    pool_lead,
    run_lead_session,
    xcorr_lags,
)
from allen_circuit_dynamics.synth import LeadSynthConfig, make_synthetic_lead


def test_xcorr_lag_sign_matches_direct_slicing():
    rng = np.random.default_rng(1)
    b = rng.normal(size=2000)
    a = np.roll(b, -7)[:, None]  # a[t] = b[t + 7]: neural leads by 7 -> peak at lag -7
    r = xcorr_lags(a, b, 10)
    lags = np.arange(-10, 11)
    assert lags[np.argmax(r[:, 0])] == -7
    # direct Pearson at that lag agrees to the edge-term tolerance
    direct = np.corrcoef(a[:-7, 0], b[7:])[0, 1]
    assert abs(r[3, 0] - direct) < 0.02


def test_rebin_and_population_roundtrip(tmp_path):
    sess, _ = make_synthetic_lead(LeadSynthConfig(n_bins=1200, n_units=(4, 3, 5)))
    coarse = sess.rebin(10)
    assert coarse.n_bins == 120 and coarse.bin_s == pytest.approx(0.1)
    assert coarse.counts.sum() == sess.counts.sum()
    ds = coarse.population("D1", "D2")
    assert ds.n_units == (4, 3) and ds.pop_names == ("D1", "D2")
    path = sess.save(tmp_path / "s")
    back = OptoSession.load(path)
    assert back.types() == sess.types() and np.array_equal(back.counts, sess.counts)


@pytest.mark.slow
def test_planted_lead_is_recovered_per_unit_and_pooled():
    cfg = LeadSynthConfig(n_bins=20000, n_units=(10, 6, 12), lead_bins=(10, 10), seed=3)
    sess, truth = make_synthetic_lead(cfg)
    lc = LeadConfig(n_null_unit=40, n_null_pop=3, bootstrap=200, n_untagged_ref=12)
    m = run_lead_session(sess, lc)
    for t in ("D1", "D2"):
        lags = m["by_type"][t]["coupled_peak_lags_s"]
        assert len(lags) >= 2
        assert -0.2 <= np.median(lags) <= -0.05  # planted lead = 0.10 s
    assert m["population"]["D1"]["peak_lag_bins"] < 0
    # uncoupled units are mostly not flagged
    flagged = np.array([u["coupled"] for u in m["units"]])
    assert flagged[~truth["coupled"]].mean() < 0.5
    pooled = pool_lead([m, m], lc)
    assert pooled["verdicts"]["H5a_pooled_lead"]["D1"]
    assert pooled["verdicts"]["H6_same_lead"]


def test_load_with_retries_retries_transient_and_not_permanent(monkeypatch):
    from allen_circuit_dynamics import aind_opto

    sess, _ = make_synthetic_lead(LeadSynthConfig(n_bins=200, n_units=(3, 3, 3)))
    calls = {"n": 0}

    def flaky(cfg, cache_dir=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("dropped connection")
        return sess

    monkeypatch.setattr(aind_opto, "load_aind_opto", flaky)
    cfg = aind_opto.AindOptoConfig(session="x")
    assert aind_opto.load_with_retries(cfg, sleep=lambda s: None) is sess
    assert calls["n"] == 3

    def missing(cfg, cache_dir=None):
        calls["n"] += 1
        raise KeyError("brain_region")

    calls["n"] = 0
    monkeypatch.setattr(aind_opto, "load_aind_opto", missing)
    with pytest.raises(KeyError):
        aind_opto.load_with_retries(cfg, sleep=lambda s: None)
    assert calls["n"] == 1  # permanent: no retry
