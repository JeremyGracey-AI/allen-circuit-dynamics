import json

import numpy as np

from allen_circuit_dynamics.cli import main
from allen_circuit_dynamics.experiments import (
    CouplingConfig,
    LatentConfig,
    run_coupling_experiment,
    run_latent_experiment,
)
from allen_circuit_dynamics.models.coupling import CoupledLDS, shuffle_conn
from allen_circuit_dynamics.synth import CoupledSynthConfig, make_synthetic_coupled


def test_latent_experiment_finds_behavior_in_shared_subspace(small_synth):
    ds, _ = small_synth
    cfg = LatentConfig(d_shared_grid=(0, 3), d_private_grid=(2,), n_splits=3, rrr_ranks=(1, 3, 6))
    m = run_latent_experiment(ds, cfg)
    sel = m["model_selection"]["selected"]
    assert sel["d_shared"] == 3
    dec = m["decoding"]
    assert dec["shared"]["r2_mean"] > dec["private2"]["r2_mean"] + 0.2
    assert dec["shared"]["r2_mean"] > dec["shared"]["null_r2_p95"]
    assert dec["private2"]["r2_mean"] < 0.1  # pop2-private was planted not to carry behavior
    curve = m["cross_population_prediction"]["rrr_r2_by_rank"]
    assert set(curve) == {1, 3, 6}
    assert max(curve.values()) > 0  # pop2 is predictable from pop1 at some rank


def test_coupling_recovers_planted_structure():
    syn = make_synthetic_coupled(CoupledSynthConfig(seed=3))
    m = run_coupling_experiment(
        syn["s"],
        syn["area_dims"],
        syn["conn"],
        syn["blocks"],
        syn["area_names"],
        CouplingConfig(n_shuffles=5, lam_grid=(0.0, 1000.0)),
    )
    assert m["corr_blocknorm_vs_connectivity_offdiag"] > 0.8
    assert m["unconstrained_r2"] > 0.5


def test_coupling_penalty_shrinks_unconnected_blocks():
    syn = make_synthetic_coupled(CoupledSynthConfig(seed=5))
    free = CoupledLDS(syn["area_dims"], conn=None, lam=0.0).fit(syn["s"], syn["blocks"])
    pen = CoupledLDS(syn["area_dims"], conn=syn["conn"], lam=1e4).fit(syn["s"], syn["blocks"])
    off = ~np.eye(len(syn["area_dims"]), dtype=bool)
    zero_blocks = (syn["conn"] == 0) & off
    assert zero_blocks.any()
    assert pen.block_norms()[zero_blocks].mean() < 0.2 * free.block_norms()[zero_blocks].mean()


def test_shuffle_keeps_diagonal(rng):
    c = np.array([[1, 0.2, 0.0], [0.5, 1, 0.1], [0.0, 0.9, 1]])
    s = shuffle_conn(c, rng)
    assert np.all(np.diag(s) == 1)
    assert sorted(s[~np.eye(3, dtype=bool)]) == sorted(c[~np.eye(3, dtype=bool)])


def test_cli_demo_writes_metrics_and_figures(tmp_path, monkeypatch):
    # keep it fast: shrink the grids through the config defaults is not exposed, so run as-is
    # on a small seed; the demo is the integration test.
    rc = main(["demo", "--out", str(tmp_path), "--seed", "0"])
    assert rc == 0
    m1 = json.loads((tmp_path / "latent" / "metrics.json").read_text())
    m2 = json.loads((tmp_path / "coupling" / "metrics.json").read_text())
    assert "provenance" in m1 and "variance_partition" in m1
    assert "by_lambda" in m2
    assert (tmp_path / "latent" / "decoding_by_subspace.png").exists()
    assert (tmp_path / "coupling" / "coupling_vs_lambda.png").exists()
