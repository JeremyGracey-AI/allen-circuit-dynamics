"""Experiment 4: does striatal spiking lead the motor output, per cell type?

Pre-registered in ``docs/experiment4_preregistration.md``; the numbers here
are the ones that file promises. Input is an :class:`OptoSession` from
:mod:`aind_opto` (10 ms spike counts of optotagged D1 / D2 / untagged units
and running speed), one session at a time, then a pooled pass over sessions.

Per session:

* per unit, the cross-correlation between its smoothed rate and speed at lags
  ``−L…+L`` (``lag = t_neural − t_speed``; negative = neural leads), against a
  circular-shift null of speed within CV blocks; a unit is *speed-coupled* if
  its peak |r| clears the null's 95th percentile;
* per cell type with enough units, Experiment 3's population ``lag_profile``
  (held-out ridge R² of speed at each lag) and the same for a random subset of
  untagged units as the reference population;
* optionally, the shared/private factor model on [D1, D2] at 100 ms with the
  lag profile of the shared latents (one session qualifies; exploratory).

Pooled: coupled-unit peak lags per type, bootstrap CIs of the medians and of
the D1−D2 difference, coupling fractions, and the H5–H7 verdicts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .aind_opto import OptoSession
from .experiments_force import lag_profile
from .lagged import circular_shift_within_blocks
from .models.shared_private import SharedPrivateFA
from .preprocess import apply_scaler, fit_scaler


@dataclass(frozen=True)
class LeadConfig:
    max_lag_bins: int = 100  # ±1 s at 10 ms
    smooth_sigma_bins: float = 2.5  # 25 ms at 10 ms, symmetric
    n_null_unit: int = 200
    min_units: int = 3
    n_untagged_ref: int = 50
    n_splits: int = 5
    decode_alpha: float = 1.0
    n_null_pop: int = 10
    pop_lag_step: int = 2  # population profile every 20 ms (ridge per lag is the slow part)
    fa_bin_factor: int = 10  # 100 ms for the exploratory factor model
    fa_d_shared: tuple[int, ...] = (1, 2)
    fa_d_private: tuple[int, ...] = (0, 1)
    fa_max_lag_bins: int = 15
    fa_min_units: tuple[int, int] = (3, 3)
    bootstrap: int = 1000
    seed: int = 0


# ----------------------------------------------------------------------------- per-unit lag profile


def _zscore(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0)
    sd = x.std(axis=0)
    return x / np.where(sd < 1e-12, 1.0, sd)


def xcorr_lags(a: np.ndarray, b: np.ndarray, max_lag: int) -> np.ndarray:
    """Pearson r between ``a[t]`` and ``b[t − lag]`` for lag = −L…+L.

    ``a`` is ``(n, k)`` (neural), ``b`` is ``(n,)`` (behaviour). Positive lag
    pairs a later neural sample with an earlier behavioural one; negative lag
    means the neural sample comes first. Both are z-scored over the full
    series and the product is divided by ``n``, so the edge bins a lag drops
    (≤ L of n) bias r toward zero by at most L/n. Computed as one matmul
    against a sliding-window view of the zero-padded behaviour, which is what
    makes 200 shift-nulls per session affordable.
    """
    n = a.shape[0]
    za, zb = _zscore(a.astype(float)), _zscore(b.astype(float)[:, None])[:, 0]
    L = max_lag
    bp = np.concatenate([np.zeros(L), zb, np.zeros(L)])
    win = np.lib.stride_tricks.sliding_window_view(bp, n)  # row s: bp[s : s + n], lag = L - s
    r = (win @ za) / n  # (2L+1, k), rows in descending lag
    return r[::-1]  # ascending: row i is lag -L + i


def unit_lag_profiles(
    rates: np.ndarray, speed: np.ndarray, blocks: np.ndarray, cfg: LeadConfig, rng
) -> list[dict]:
    """Per-unit peak lag, peak r and shift-null p95 of the max |r| over lags."""
    L = cfg.max_lag_bins
    r = xcorr_lags(rates, speed, L)  # (2L+1, k)
    null_max = np.empty((cfg.n_null_unit, rates.shape[1]))
    for s in range(cfg.n_null_unit):
        sp = circular_shift_within_blocks(speed, blocks, rng, min_shift=L + 1)
        null_max[s] = np.abs(xcorr_lags(rates, sp, L)).max(axis=0)
    p95 = np.percentile(null_max, 95, axis=0)
    lags = np.arange(-L, L + 1)
    out = []
    for k in range(rates.shape[1]):
        i = int(np.argmax(np.abs(r[:, k])))
        out.append(
            {
                "peak_lag_bins": int(lags[i]),
                "peak_r": float(r[i, k]),
                "r_at_zero": float(r[L, k]),
                "null_p95_abs_r": float(p95[k]),
                "coupled": bool(abs(r[i, k]) > p95[k]),
            }
        )
    return out


# ----------------------------------------------------------------------------- per session


def _pop_profile(
    feat: np.ndarray, speed: np.ndarray, blocks: np.ndarray, cfg: LeadConfig, seed: int
) -> dict:
    """Experiment 3's lag profile on a coarser lag grid (every ``pop_lag_step`` bins)."""
    step = max(1, cfg.pop_lag_step)
    # Decimate time by `step` so one profile bin == `step` original bins.
    n = (feat.shape[0] // step) * step
    f = feat[:n].reshape(n // step, step, -1).mean(axis=1)
    s = speed[:n].reshape(n // step, step).mean(axis=1)
    b = blocks[:n:step]
    prof = lag_profile(
        f,
        s,
        b,
        cfg.max_lag_bins // step,
        n_splits=cfg.n_splits,
        alpha=cfg.decode_alpha,
        n_null=cfg.n_null_pop,
        seed=seed,
    )
    prof["lags_bins"] = [int(v * step) for v in prof["lags_bins"]]
    prof["peak_lag_bins"] = int(prof["peak_lag_bins"] * step)
    prof["lag_step_bins"] = step
    return prof


def run_lead_session(sess: OptoSession, cfg: LeadConfig | None = None) -> dict:
    """Experiment 4 on one session."""
    cfg = cfg or LeadConfig()
    rng = np.random.default_rng(cfg.seed)
    sig = cfg.smooth_sigma_bins
    counts = sess.counts.astype(float)
    rates = gaussian_filter1d(counts, sigma=sig, axis=0, mode="nearest") / sess.bin_s
    speed = gaussian_filter1d(sess.speed.astype(float), sigma=sig, mode="nearest")
    blocks = sess.blocks

    per_unit = unit_lag_profiles(rates, speed, blocks, cfg, rng)
    for k, rec in enumerate(per_unit):
        rec["unit_id"] = int(sess.unit_id[k])
        rec["type"] = str(sess.unit_type[k])
        rec["rate_hz"] = float(sess.counts[:, k].mean() / sess.bin_s)
        rec["peak_lag_s"] = rec["peak_lag_bins"] * sess.bin_s

    pops: dict[str, dict] = {}
    groups: dict[str, np.ndarray] = {}
    for t in sorted(sess.types()):
        if t == "untagged":
            continue
        idx = sess.units_of(t)
        if idx.size >= cfg.min_units:
            groups[t] = idx
    ut = sess.units_of("untagged")
    if ut.size >= cfg.min_units:
        groups["untagged_ref"] = np.sort(
            rng.choice(ut, min(cfg.n_untagged_ref, ut.size), replace=False)
        )
    for name, idx in groups.items():
        z = _zscore(rates[:, idx])
        prof = _pop_profile(z, speed, blocks, cfg, cfg.seed)
        prof["peak_lag_s"] = prof["peak_lag_bins"] * sess.bin_s
        # Post hoc, not pre-registered: the mean z-rate of the group cross-correlated with
        # speed, same null as the per-unit test. Added after the first real session showed
        # the held-out ridge profile is dominated by block-to-block changes in how much the
        # mouse runs (negative R² at every lag), which says nothing about timing.
        xc = unit_lag_profiles(z.mean(axis=1, keepdims=True), speed, blocks, cfg, rng)[0]
        xc["peak_lag_s"] = xc["peak_lag_bins"] * sess.bin_s
        xc["r"] = [
            float(v)
            for v in xcorr_lags(z.mean(axis=1, keepdims=True), speed, cfg.max_lag_bins)[:, 0]
        ]
        pops[name] = {
            "n_units": int(idx.size),
            "unit_ids": sess.unit_id[idx].tolist(),
            "mean_rate_xcorr_post_hoc": xc,
        } | prof

    fa = None
    n1, n2 = sess.units_of("D1").size, sess.units_of("D2").size
    if n1 >= cfg.fa_min_units[0] and n2 >= cfg.fa_min_units[1]:
        fa = _exploratory_fa(sess, cfg)

    by_type = {
        t: {
            "n": int(sum(1 for u in per_unit if u["type"] == t)),
            "n_coupled": int(sum(1 for u in per_unit if u["type"] == t and u["coupled"])),
            "coupled_peak_lags_s": [
                u["peak_lag_s"] for u in per_unit if u["type"] == t and u["coupled"]
            ],
        }
        for t in sorted(sess.types())
    }
    return {
        "experiment": "lead",
        "session": sess.meta.get("session"),
        "subject_id": sess.meta.get("subject_id"),
        "genotype": sess.meta.get("genotype"),
        "epoch": sess.meta.get("epoch"),
        "epoch_window_s": sess.meta.get("epoch_window_s"),
        "n_bins": sess.n_bins,
        "bin_s": sess.bin_s,
        "cell_types": sess.types(),
        "by_type": by_type,
        "units": per_unit,
        "population": pops,
        "exploratory_fa": fa,
        "config": asdict(cfg),
        "dataset_meta": {k: v for k, v in sess.meta.items() if k not in ("nwb", "asset")},
    }


def _exploratory_fa(sess: OptoSession, cfg: LeadConfig) -> dict:
    """Experiment 3's model on [D1, D2] at a coarser bin, with the shared-latent lag profile."""
    from sklearn.model_selection import GroupKFold

    coarse = sess.rebin(cfg.fa_bin_factor)
    ds = coarse.population("D1", "D2")
    x1, x2, speed, blocks = ds.x1, ds.x2, ds.behavior, ds.blocks
    gkf = GroupKFold(n_splits=min(cfg.n_splits, np.unique(blocks).size))
    folds = list(gkf.split(x1, groups=blocks))
    grid = []
    for d_s in cfg.fa_d_shared:
        for d_p in cfg.fa_d_private:
            if d_s + d_p >= min(x1.shape[1], x2.shape[1]):
                continue
            lls = []
            for tr, te in folds:
                s1, s2 = fit_scaler(x1[tr]), fit_scaler(x2[tr])
                m = SharedPrivateFA(d_shared=d_s, d_private=(d_p, d_p), seed=cfg.seed).fit(
                    apply_scaler(x1[tr], s1), apply_scaler(x2[tr], s2)
                )
                lls.append(m.score(apply_scaler(x1[te], s1), apply_scaler(x2[te], s2)))
            grid.append({"d_shared": d_s, "d_private": d_p, "heldout_loglik": float(np.mean(lls))})
    best = max(grid, key=lambda g: g["heldout_loglik"])
    s1, s2 = fit_scaler(x1), fit_scaler(x2)
    a, b = apply_scaler(x1, s1), apply_scaler(x2, s2)
    model = SharedPrivateFA(
        d_shared=best["d_shared"], d_private=(best["d_private"], best["d_private"]), seed=cfg.seed
    ).fit(a, b)
    lat = model.transform(a, b)
    vp = model.variance_partition()
    kw = dict(n_splits=cfg.n_splits, alpha=cfg.decode_alpha, n_null=cfg.n_null_pop, seed=cfg.seed)
    prof = lag_profile(lat["shared"], speed, blocks, cfg.fa_max_lag_bins, **kw)
    prof["peak_lag_s"] = prof["peak_lag_bins"] * coarse.bin_s
    return {
        "note": "exploratory: one session, few D2 units; never generalised",
        "bin_s": coarse.bin_s,
        "n_units": [int(x1.shape[1]), int(x2.shape[1])],
        "selected": best,
        "model_selection_grid": grid,
        "variance_partition": {
            pop: {"shared": v["shared"], "private": v["private"], "independent": v["noise"]}
            for pop, v in (("D1", vp["pop1"]), ("D2", vp["pop2"]))
        },
        "shared_lag_profile": prof,
    }


# ----------------------------------------------------------------------------- pooled


def _boot_median(vals: np.ndarray, n: int, rng) -> tuple[float, list[float]]:
    if vals.size == 0:
        return float("nan"), [float("nan"), float("nan")]
    meds = np.array([np.median(rng.choice(vals, vals.size, replace=True)) for _ in range(n)])
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return float(np.median(vals)), [float(lo), float(hi)]


def pool_lead(results: list[dict], cfg: LeadConfig | None = None) -> dict:
    """Pool per-session results and score H5–H7 against the pre-registered thresholds."""
    cfg = cfg or LeadConfig()
    rng = np.random.default_rng(cfg.seed)
    types = ("D1", "D2")
    pooled: dict[str, dict] = {}
    for t in types:
        lags, n, n_c, sessions = [], 0, 0, 0
        for r in results:
            bt = r["by_type"].get(t)
            if not bt or bt["n"] < cfg.min_units:
                continue
            sessions += 1
            n += bt["n"]
            n_c += bt["n_coupled"]
            lags.extend(bt["coupled_peak_lags_s"])
        arr = np.asarray(lags, dtype=float)
        med, ci = _boot_median(arr, cfg.bootstrap, rng)
        pop = [
            r["population"][t]
            for r in results
            if t in r["population"] and r["population"][t]["n_units"] >= cfg.min_units
        ]
        neg_above = sum(1 for p in pop if p["peak_lag_bins"] < 0 and p["peak_above_null"])
        xc = [p["mean_rate_xcorr_post_hoc"] for p in pop if "mean_rate_xcorr_post_hoc" in p]
        xc_neg = sum(1 for x in xc if x["peak_lag_bins"] < 0 and x["coupled"])
        pooled[t] = {
            "n_sessions": sessions,
            "n_units": n,
            "n_coupled": n_c,
            "coupled_fraction": (n_c / n) if n else float("nan"),
            "coupled_peak_lags_s": lags,
            "median_lag_s": med,
            "median_ci95_s": ci,
            "population_profiles": len(pop),
            "population_negative_above_null": neg_above,
            "population_peak_lags_s": [p["peak_lag_s"] for p in pop],
            "post_hoc_mean_rate_xcorr": {
                "n_profiles": len(xc),
                "negative_and_coupled": xc_neg,
                "peak_lags_s": [x["peak_lag_s"] for x in xc],
            },
        }
    d1 = np.asarray(pooled["D1"]["coupled_peak_lags_s"])
    d2 = np.asarray(pooled["D2"]["coupled_peak_lags_s"])
    if d1.size and d2.size:
        diffs = np.array(
            [
                np.median(rng.choice(d1, d1.size, replace=True))
                - np.median(rng.choice(d2, d2.size, replace=True))
                for _ in range(cfg.bootstrap)
            ]
        )
        diff = {
            "median_diff_s": float(np.median(d1) - np.median(d2)),
            "ci95_s": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        }
    else:
        diff = {"median_diff_s": float("nan"), "ci95_s": [float("nan"), float("nan")]}

    def h5a(t):
        m, (_, hi) = pooled[t]["median_lag_s"], pooled[t]["median_ci95_s"]
        return bool(-0.5 <= m <= -0.05 and hi < 0)

    def h5b(t):
        k, n = pooled[t]["population_negative_above_null"], pooled[t]["population_profiles"]
        return bool(n > 0 and k >= np.ceil(2 * n / 3))

    lo, hi = diff["ci95_s"]
    verdicts = {
        "H5a_pooled_lead": {t: h5a(t) for t in types},
        "H5b_population_lead": {t: h5b(t) for t in types},
        "H6_same_lead": bool(lo >= -0.1 and hi <= 0.1) if np.isfinite(lo) else False,
        "H7_coupled_fraction": {t: bool(pooled[t]["coupled_fraction"] >= 0.30) for t in types},
    }
    verdicts["H5a_supported"] = all(verdicts["H5a_pooled_lead"].values())
    verdicts["H5b_supported"] = all(verdicts["H5b_population_lead"].values())
    verdicts["H7_supported"] = all(verdicts["H7_coupled_fraction"].values())
    untagged = [
        r["population"]["untagged_ref"]["peak_lag_s"]
        for r in results
        if "untagged_ref" in r["population"]
    ]
    return {
        "experiment": "lead_pooled",
        "n_sessions": len(results),
        "per_type": pooled,
        "D1_minus_D2": diff,
        "untagged_reference_peak_lags_s": untagged,
        "verdicts": verdicts,
        "thresholds": {
            "H5a": "median coupled-unit lag in [-0.5, -0.05] s, CI95 upper < 0, both types",
            "H5b": "population peak negative and above null in >= 2/3 of eligible sessions, both",
            "H6": "bootstrap CI95 of median(D1) - median(D2) within +-0.1 s",
            "H7": "coupled fraction >= 0.30, both types",
        },
        "config": asdict(cfg),
    }
