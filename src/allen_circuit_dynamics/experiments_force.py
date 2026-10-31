"""Experiment 3: D1 vs D2 striatal neurons, shared / private covariance, and the lever.

Pre-registered in ``docs/experiment3_preregistration.md``; the numbers here
are the ones that file promises. The dataset is any :class:`PopulationDataset`
whose ``extras`` carry a ``context`` label (1 = cued, 0 = self-paced) and a
``motion`` trace per bin; :mod:`klee_data` builds one.

Per subsample draw (D1 and D2 cut to equal size) and per context (all / cued /
self-paced):

* model selection by held-out log-likelihood on the pre-registered grid;
* variance partition, reported three ways (shared / private / independent);
* held-out decoding of the lever from each subspace against a circular-shift
  null, raw and after regressing video motion energy out of the populations
  inside the training fold;
* a lag profile: held-out decoding of the lever at ``lag = t_neural - t_lever``
  from -L..+L bins, so a peak at a negative lag means the neural signal leads;
* reduced-rank cross-prediction in both directions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from .dataset import PopulationDataset
from .decode import DecodeResult, _r2, decode_cv
from .lagged import lag_pairs
from .models.baselines import reduced_rank_regression
from .models.shared_private import SharedPrivateFA
from .preprocess import apply_scaler, fit_scaler


@dataclass(frozen=True)
class ForceConfig:
    d_shared_grid: tuple[int, ...] = (1, 2, 3, 4, 6, 8)
    d_private_grid: tuple[int, ...] = (0, 1, 2, 4)
    n_splits: int = 5
    n_draws: int = 10
    max_lag_bins: int = 15  # +-1.5 s at 100 ms
    motion_lags: tuple[int, ...] = (-2, -1, 0, 1, 2)
    rrr_ranks: tuple[int, ...] = (1, 2, 4, 8)
    decode_alpha: float = 1.0
    n_null: int = 50
    n_null_lag: int = 10
    contexts: tuple[str, ...] = ("all", "cued", "selfpaced")
    seed: int = 0


# ----------------------------------------------------------------------------- helpers


def _lagged_basis(v: np.ndarray, blocks: np.ndarray, lags: tuple[int, ...]) -> np.ndarray:
    """Columns ``v[t + lag]`` for each lag, zero where the shift leaves the block."""
    cols = []
    n = v.size
    t = np.arange(n)
    for lag in lags:
        j = np.clip(t + lag, 0, n - 1)
        ok = (t + lag >= 0) & (t + lag < n) & (blocks[j] == blocks)
        cols.append(np.where(ok, v[j], 0.0))
    return np.column_stack(cols)


def _residualize(x_tr: np.ndarray, x_te: np.ndarray, c_tr: np.ndarray, c_te: np.ndarray):
    """Regress covariates out of ``x`` with coefficients fit on the training rows only."""
    c_tr1 = np.column_stack([c_tr, np.ones(c_tr.shape[0])])
    c_te1 = np.column_stack([c_te, np.ones(c_te.shape[0])])
    beta, *_ = np.linalg.lstsq(c_tr1, x_tr, rcond=None)
    return x_tr - c_tr1 @ beta, x_te - c_te1 @ beta


def decode_cv_residualized(
    features: np.ndarray,
    behavior: np.ndarray,
    blocks: np.ndarray,
    covariates: np.ndarray,
    n_splits: int = 5,
    alpha: float = 1.0,
    n_null: int = 50,
    seed: int = 0,
) -> DecodeResult:
    """``decode_cv`` after regressing ``covariates`` out of the features inside each fold."""
    rng = np.random.default_rng(seed)
    gkf = GroupKFold(n_splits=min(n_splits, np.unique(blocks).size))
    r2s, nulls = [], []
    for tr, te in gkf.split(features, behavior, groups=blocks):
        f_tr, f_te = _residualize(features[tr], features[te], covariates[tr], covariates[te])
        model = Ridge(alpha=alpha).fit(f_tr, behavior[tr])
        yhat = model.predict(f_te)
        r2s.append(_r2(behavior[te], yhat))
        n_te = te.size
        for _ in range(n_null):
            shift = int(rng.integers(n_te // 10, n_te - n_te // 10))
            nulls.append(_r2(np.roll(behavior[te], shift), yhat))
    nulls_arr = np.asarray(nulls)
    return DecodeResult(
        r2_mean=float(np.mean(r2s)),
        r2_std=float(np.std(r2s)),
        r2_folds=[float(v) for v in r2s],
        null_r2_mean=float(np.mean(nulls_arr)),
        null_r2_p95=float(np.percentile(nulls_arr, 95)),
        n_features=int(features.shape[1]),
    )


def lag_profile(
    features: np.ndarray,
    behavior: np.ndarray,
    blocks: np.ndarray,
    max_lag: int,
    **kw,
) -> dict:
    """Held-out R² of ``behavior[j]`` from ``features[i]``; lag = i - j (< 0: neural leads)."""
    lags = list(range(-max_lag, max_lag + 1))
    r2, p95 = [], []
    for lag in lags:
        # i = neural index, j = lever index, lag = i - j  ->  j = i - lag
        i, j = lag_pairs(blocks, -lag)
        res = decode_cv(features[i], behavior[j], blocks[i], **kw)
        r2.append(res.r2_mean)
        p95.append(res.null_r2_p95)
    r2a = np.asarray(r2)
    k = int(np.argmax(r2a))
    return {
        "lags_bins": lags,
        "r2": [float(v) for v in r2],
        "null_p95": [float(v) for v in p95],
        "peak_lag_bins": int(lags[k]),
        "peak_r2": float(r2a[k]),
        "peak_above_null": bool(r2a[k] > p95[k]),
    }


def _select(x1, x2, blocks, cfg: ForceConfig) -> tuple[dict, list[dict]]:
    gkf = GroupKFold(n_splits=min(cfg.n_splits, np.unique(blocks).size))
    folds = list(gkf.split(x1, groups=blocks))
    grid = []
    for ds_ in cfg.d_shared_grid:
        for dp in cfg.d_private_grid:
            lls = []
            for tr, te in folds:
                s1, s2 = fit_scaler(x1[tr]), fit_scaler(x2[tr])
                m = SharedPrivateFA(d_shared=ds_, d_private=(dp, dp), seed=cfg.seed).fit(
                    apply_scaler(x1[tr], s1), apply_scaler(x2[tr], s2)
                )
                lls.append(m.score(apply_scaler(x1[te], s1), apply_scaler(x2[te], s2)))
            grid.append({"d_shared": ds_, "d_private": dp, "heldout_loglik": float(np.mean(lls))})
    return max(grid, key=lambda g: g["heldout_loglik"]), grid


def _one(x1, x2, lever, blocks, motion, cfg: ForceConfig, seed: int) -> dict:
    """Everything for one (draw, context)."""
    best, grid = _select(x1, x2, blocks, cfg)
    s1, s2 = fit_scaler(x1), fit_scaler(x2)
    a, b = apply_scaler(x1, s1), apply_scaler(x2, s2)
    model = SharedPrivateFA(
        d_shared=best["d_shared"], d_private=(best["d_private"], best["d_private"]), seed=cfg.seed
    ).fit(a, b)
    lat = model.transform(a, b)
    vp = model.variance_partition()
    partition = {
        pop: {
            "shared": v["shared"],
            "private": v["private"],
            "independent": v["noise"],
            "private_plus_independent_athalye2017": v["private"] + v["noise"],
        }
        for pop, v in (("D1", vp["pop1"]), ("D2", vp["pop2"]))
    }

    sets = {
        "shared": lat["shared"],
        "private_D1": lat["private1"],
        "private_D2": lat["private2"],
        "private_both": np.hstack([lat["private1"], lat["private2"]]),
        "all_latents": np.hstack([lat["shared"], lat["private1"], lat["private2"]]),
        "raw_D1": a,
        "raw_D2": b,
        "raw_both": np.hstack([a, b]),
    }
    kw = dict(n_splits=cfg.n_splits, alpha=cfg.decode_alpha, n_null=cfg.n_null, seed=seed)
    dec = {k: asdict(decode_cv(v, lever, blocks, **kw)) for k, v in sets.items() if v.shape[1]}
    cov = _lagged_basis(motion, blocks, cfg.motion_lags)
    dec_motion = {
        k: asdict(decode_cv_residualized(v, lever, blocks, cov, **kw))
        for k, v in sets.items()
        if v.shape[1]
    }
    lkw = dict(n_splits=cfg.n_splits, alpha=cfg.decode_alpha, n_null=cfg.n_null_lag, seed=seed)
    lags = {
        k: lag_profile(sets[k], lever, blocks, cfg.max_lag_bins, **lkw)
        for k in ("shared", "private_D1", "private_D2")
        if sets[k].shape[1]
    }

    gkf = GroupKFold(n_splits=min(cfg.n_splits, np.unique(blocks).size))
    rrr = {
        "D2_from_D1": {r: [] for r in cfg.rrr_ranks},
        "D1_from_D2": {r: [] for r in cfg.rrr_ranks},
    }
    for tr, te in gkf.split(a, groups=blocks):
        ranks = [r for r in cfg.rrr_ranks if r <= min(a.shape[1], b.shape[1])]
        for r, v in reduced_rank_regression(a[tr], b[tr], a[te], b[te], ranks).items():
            rrr["D2_from_D1"][r].append(v)
        for r, v in reduced_rank_regression(b[tr], a[tr], b[te], a[te], ranks).items():
            rrr["D1_from_D2"][r].append(v)
    rrr_out = {d: {int(r): float(np.mean(v)) for r, v in rv.items() if v} for d, rv in rrr.items()}

    return {
        "n_bins": int(a.shape[0]),
        "n_units": [int(a.shape[1]), int(b.shape[1])],
        "selected": best,
        "model_selection_grid": grid,
        "variance_partition": partition,
        "decoding": dec,
        "decoding_motion_regressed": dec_motion,
        "lag_profile": lags,
        "rrr_r2_by_rank": rrr_out,
    }


def _mask_context(ds: PopulationDataset, name: str) -> np.ndarray:
    ctx = ds.extras["context"]
    if name == "all":
        return np.ones(ds.n_bins, dtype=bool)
    if name == "cued":
        return ctx == 1
    if name == "selfpaced":
        return ctx == 0
    raise ValueError(name)


def run_force_experiment(ds: PopulationDataset, cfg: ForceConfig | None = None) -> dict:
    """Experiment 3 on one mouse: all draws, all contexts, plus a per-context summary."""
    cfg = cfg or ForceConfig()
    rng = np.random.default_rng(cfg.seed)
    n1, n2 = ds.n_units
    n = min(n1, n2)
    draws = [
        {
            "D1": np.sort(rng.choice(n1, n, replace=False)),
            "D2": np.sort(rng.choice(n2, n, replace=False)),
        }
        for _ in range(cfg.n_draws)
    ]
    motion = ds.extras["motion"]
    results: dict[str, list[dict]] = {c: [] for c in cfg.contexts}
    for d, idx in enumerate(draws):
        x1, x2 = ds.x1[:, idx["D1"]], ds.x2[:, idx["D2"]]
        for c in cfg.contexts:
            m = _mask_context(ds, c)
            if m.sum() < 200:
                continue
            r = _one(x1[m], x2[m], ds.behavior[m], ds.blocks[m], motion[m], cfg, seed=cfg.seed + d)
            r["draw"] = d
            results[c].append(r)
    return {
        "experiment": "force",
        "dataset": ds.summary()
        | {"n_rois": ds.meta.get("n_rois"), "animal": ds.meta.get("animal")},
        "count_matched_n": n,
        "draws": [{k: v.tolist() for k, v in d.items()} for d in draws],
        "per_context": results,
        "summary": {c: summarize_context(v) for c, v in results.items() if v},
        "config": asdict(cfg),
    }


def _ms(vals) -> dict:
    v = np.asarray(vals, dtype=float)
    return {"mean": float(v.mean()), "sd": float(v.std()), "n": int(v.size)}


def summarize_context(runs: list[dict]) -> dict:
    """Mean ± SD across subsample draws of the numbers the pre-registration names."""
    out: dict = {
        "selected_d_shared": _ms([r["selected"]["d_shared"] for r in runs]),
        "selected_d_private": _ms([r["selected"]["d_private"] for r in runs]),
        "variance_partition": {
            pop: {
                k: _ms([r["variance_partition"][pop][k] for r in runs])
                for k in runs[0]["variance_partition"][pop]
            }
            for pop in ("D1", "D2")
        },
        "decoding_r2": {},
        "decoding_null_p95": {},
        "decoding_motion_regressed_r2": {},
        "lag_peak_bins": {},
        "lag_peak_above_null_frac": {},
    }
    for k in runs[0]["decoding"]:
        out["decoding_r2"][k] = _ms(
            [r["decoding"][k]["r2_mean"] for r in runs if k in r["decoding"]]
        )
        out["decoding_null_p95"][k] = _ms(
            [r["decoding"][k]["null_r2_p95"] for r in runs if k in r["decoding"]]
        )
    for k in runs[0]["decoding_motion_regressed"]:
        out["decoding_motion_regressed_r2"][k] = _ms(
            [
                r["decoding_motion_regressed"][k]["r2_mean"]
                for r in runs
                if k in r["decoding_motion_regressed"]
            ]
        )
    for k in runs[0]["lag_profile"]:
        out["lag_peak_bins"][k] = _ms(
            [r["lag_profile"][k]["peak_lag_bins"] for r in runs if k in r["lag_profile"]]
        )
        out["lag_peak_above_null_frac"][k] = float(
            np.mean([r["lag_profile"][k]["peak_above_null"] for r in runs if k in r["lag_profile"]])
        )
    out["lag_profile_mean"] = {
        k: {
            "lags_bins": runs[0]["lag_profile"][k]["lags_bins"],
            "r2": np.mean(
                [r["lag_profile"][k]["r2"] for r in runs if k in r["lag_profile"]], axis=0
            ).tolist(),
            "null_p95": np.mean(
                [r["lag_profile"][k]["null_p95"] for r in runs if k in r["lag_profile"]], axis=0
            ).tolist(),
        }
        for k in runs[0]["lag_profile"]
    }
    out["rrr_r2_by_rank"] = {
        d: {
            r: _ms([run["rrr_r2_by_rank"][d][r] for run in runs])
            for r in runs[0]["rrr_r2_by_rank"][d]
        }
        for d in runs[0]["rrr_r2_by_rank"]
    }
    return out
