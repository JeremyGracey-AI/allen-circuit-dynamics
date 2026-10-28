import numpy as np
import pytest

from allen_circuit_dynamics.synth import SynthConfig, make_synthetic


@pytest.fixture(scope="session")
def small_synth():
    """Low-gain, low-rate regime where the linear-Gaussian model is well specified."""
    cfg = SynthConfig(
        n_bins=3000,
        n_units=(30, 24),
        d_shared=3,
        d_private=(2, 2),
        base_rate_hz=8.0,
        shared_gain=1.0,
        private_gain=0.7,
        seed=1,
    )
    return make_synthetic(cfg)


@pytest.fixture
def rng():
    return np.random.default_rng(123)
