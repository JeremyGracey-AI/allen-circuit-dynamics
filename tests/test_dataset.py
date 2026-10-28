import numpy as np
import pytest

from allen_circuit_dynamics.dataset import PopulationDataset


def test_shape_validation_rejects_mismatch():
    with pytest.raises(ValueError, match="bins"):
        PopulationDataset(
            x1=np.zeros((10, 3)),
            x2=np.zeros((9, 2)),
            behavior=np.zeros(10),
            blocks=np.zeros(10, dtype=int),
            bin_s=0.05,
        )


def test_nonfinite_rejected():
    x1 = np.zeros((10, 3))
    x1[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        PopulationDataset(
            x1=x1,
            x2=np.zeros((10, 2)),
            behavior=np.zeros(10),
            blocks=np.zeros(10, dtype=int),
            bin_s=0.05,
        )


def test_save_load_roundtrip(tmp_path, small_synth):
    ds, _ = small_synth
    p = ds.save(tmp_path / "ds")
    back = PopulationDataset.load(p)
    assert back.n_bins == ds.n_bins
    assert back.pop_names == ds.pop_names
    np.testing.assert_allclose(back.x1, ds.x1)
    np.testing.assert_allclose(back.behavior, ds.behavior)
    assert back.meta["source"] == "synthetic"
