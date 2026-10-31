
from allen_circuit_dynamics.experiments_force import ForceConfig, lag_profile, run_force_experiment
from allen_circuit_dynamics.synth import ForceSynthConfig, make_synthetic_force


def test_lag_profile_recovers_planted_lead():
    ds, truth = make_synthetic_force(ForceSynthConfig(seed=1, lead_bins=4, n_bins=4000))
    prof = lag_profile(truth["z"], ds.behavior, ds.blocks, max_lag=8, n_null=5)
    # lag = t_neural - t_lever; the lever follows the shared factors, so the peak is negative
    assert prof["peak_lag_bins"] == -4
    assert prof["peak_above_null"]


def test_force_experiment_count_matches_and_finds_shared():
    ds, truth = make_synthetic_force(ForceSynthConfig(seed=2, n_bins=3000))
    cfg = ForceConfig(
        n_draws=2,
        contexts=("all", "cued"),
        d_shared_grid=(1, 3),
        d_private_grid=(0, 2),
        max_lag_bins=5,
        n_null=10,
        n_null_lag=5,
    )
    m = run_force_experiment(ds, cfg)
    assert m["count_matched_n"] == 40
    for d in m["draws"]:
        assert len(d["D1"]) == len(d["D2"]) == 40
    s = m["summary"]["all"]
    assert s["decoding_r2"]["shared"]["mean"] > s["decoding_r2"]["private_both"]["mean"]
    assert s["decoding_r2"]["shared"]["mean"] > s["decoding_null_p95"]["shared"]["mean"]
    assert s["lag_peak_bins"]["shared"]["mean"] == -4
    assert "cued" in m["summary"]
