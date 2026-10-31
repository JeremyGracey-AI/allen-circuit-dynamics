import numpy as np

from allen_circuit_dynamics.lagged import align, circular_shift_within_blocks, lag_pairs


def test_lag_pairs_never_cross_blocks():
    blocks = np.repeat([0, 1, 2], 10)
    for lag in (-4, -1, 0, 1, 4):
        i, j = lag_pairs(blocks, lag)
        assert np.all(j - i == lag)
        assert np.all(blocks[i] == blocks[j])
        # each block of 10 yields 10 - |lag| pairs
        assert i.size == 3 * (10 - abs(lag))


def test_align_recovers_planted_lag():
    rng = np.random.default_rng(0)
    blocks = np.repeat(np.arange(6), 200)
    z = rng.normal(size=(1200, 3))
    x1 = z + 0.1 * rng.normal(size=z.shape)
    x2 = np.roll(z, 3, axis=0) + 0.1 * rng.normal(size=z.shape)  # x2[t] = z[t-3]: pop2 lags by 3
    best = max(
        range(-6, 7),
        key=lambda lag: np.corrcoef(*(a.ravel() for a in align(x1, x2, blocks, lag)[:2]))[0, 1],
    )
    assert best == 3


def test_circular_shift_keeps_block_contents():
    rng = np.random.default_rng(1)
    blocks = np.repeat([0, 1], 50)
    x = rng.normal(size=(100, 4))
    y = circular_shift_within_blocks(x, blocks, rng)
    for b in (0, 1):
        assert np.allclose(np.sort(x[blocks == b], axis=0), np.sort(y[blocks == b], axis=0))
    assert not np.allclose(x, y)
