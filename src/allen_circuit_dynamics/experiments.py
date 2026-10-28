"""The two experiments, as functions from a dataset to a metrics dict.

Both are deterministic given the seed, run on synthetic data in seconds, and
write nothing — the CLI owns files. Every number that lands in ``metrics.json``
comes out of here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.model_selection import GroupKFold

from .dataset import PopulationDataset
from .decode import decode_all
from .models.baselines import cca_shared, per_population_fa, reduced_rank_regression
from .models.coupling import CoupledLDS, shuffle_conn
from .models.shared_private import SharedPrivateFA
from .preprocess import PreprocessConfig, apply_scaler, filter_units, fit_scaler, transform_counts


@dataclass(frozen=True)
class LatentConfig:
    d_shared_grid: tuple[int, ...] = (0, 1, 2, 3, 4, 6)
    d_private_grid: tuple[int, ...] = (0, 1, 2, 4)
    n_splits: int = 5
    rrr_ranks: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12)
    decode_alpha: float = 1.0
    seed: int = 0


def _prep(ds: PopulationDataset, pcfg: PreprocessConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    x1, info1 = filter_units(ds.x1, ds.bin_s, pcfg.min_rate_hz)
    x2, info2 = filter_units(ds.x2, ds.bin_s, pcfg.min_rate_hz)
    x1, steps = transform_counts(x1, pcfg)
    x2, _ = transform_counts(x2, pcfg)
    return x1, x2, {"pop1": info1, "pop2": info2, "steps": steps}


def run_latent_experiment(
    ds: PopulationDataset,
    cfg: LatentConfig | None = None,
    pcfg: PreprocessConfig | None = None,
) -> dict:
    """Experiment 1: shared vs. private structure and which subspace carries behavior."""
    cfg = cfg or LatentConfig()
    pcfg = pcfg or PreprocessConfig()
    x1, x2, prep_info = _prep(ds, pcfg)
    blocks, behavior = ds.blocks, ds.behavior
    gkf = GroupKFold(n_splits=min(cfg.n_splits, ds.n_blocks))
    folds = list(gkf.split(x1, groups=blocks))

    # --- model selection on held-out log-likelihood -----------------------
    grid: list[dict] = []
    for ds_ in cfg.d_shared_grid:
        for dp in cfg.d_private_grid:
            if ds_ == 0 and dp == 0:
                continue
            lls = []
            for tr, te in folds:
                s1, s2 = fit_scaler(x1[tr]), fit_scaler(x2[tr])
                a_tr, b_tr = apply_scaler(x1[tr], s1), apply_scaler(x2[tr], s2)
                a_te, b_te = apply_scaler(x1[te], s1), apply_scaler(x2[te], s2)
                m = SharedPrivateFA(d_shared=ds_, d_private=(dp, dp), seed=cfg.seed).fit(a_tr, b_tr)
                lls.append(m.score(a_te, b_te))
            grid.append(
                {
                    "d_shared": ds_,
                    "d_private": dp,
                    "heldout_loglik": float(np.mean(lls)),
                    "heldout_loglik_std": float(np.std(lls)),
                }
            )
    best = max(grid, key=lambda g: g["heldout_loglik"])

    # --- refit the selected model on everything ---------------------------
    s1, s2 = fit_scaler(x1), fit_scaler(x2)
    a, b = apply_scaler(x1, s1), apply_scaler(x2, s2)
    model = SharedPrivateFA(
        d_shared=best["d_shared"], d_private=(best["d_private"], best["d_private"]), seed=cfg.seed
    ).fit(a, b)
    partition = model.variance_partition()
    latents = model.transform(a, b)

    # --- cross-population prediction: model vs. RRR -----------------------
    cross = {"model_r2": [], "rrr": {r: [] for r in cfg.rrr_ranks}, "cca": []}
    fa_ll = {"pop1": [], "pop2": [], "joint_model": []}
    for tr, te in folds:
        s1f, s2f = fit_scaler(x1[tr]), fit_scaler(x2[tr])
        a_tr, b_tr = apply_scaler(x1[tr], s1f), apply_scaler(x2[tr], s2f)
        a_te, b_te = apply_scaler(x1[te], s1f), apply_scaler(x2[te], s2f)
        mf = SharedPrivateFA(
            d_shared=best["d_shared"],
            d_private=(best["d_private"], best["d_private"]),
            seed=cfg.seed,
        ).fit(a_tr, b_tr)
        pred = mf.predict_cross(a_te)
        ss_res = np.sum((b_te - pred) ** 2)
        ss_tot = np.sum((b_te - b_te.mean(0)) ** 2)
        cross["model_r2"].append(float(1 - ss_res / ss_tot))
        ranks = [r for r in cfg.rrr_ranks if r <= min(a_tr.shape[1], b_tr.shape[1])]
        for r, v in reduced_rank_regression(a_tr, b_tr, a_te, b_te, ranks).items():
            cross["rrr"][r].append(v)
        cross["cca"].append(cca_shared(a_tr, b_tr, a_te, b_te, max(best["d_shared"], 1)))
        k_pop = max(best["d_shared"] + best["d_private"], 1)
        fa_ll["pop1"].append(per_population_fa(a_tr, a_te, k_pop)["heldout_loglik_per_bin"])
        fa_ll["pop2"].append(per_population_fa(b_tr, b_te, k_pop)["heldout_loglik_per_bin"])
        fa_ll["joint_model"].append(mf.score(a_te, b_te))

    rrr_curve = {int(r): float(np.mean(v)) for r, v in cross["rrr"].items() if v}
    cca_mean = np.mean([c["heldout_canonical_corr"] for c in cross["cca"]], axis=0).tolist()

    # --- behavior decoding ------------------------------------------------
    dec = decode_all(
        latents,
        a,
        b,
        behavior,
        blocks,
        n_splits=cfg.n_splits,
        alpha=cfg.decode_alpha,
        seed=cfg.seed,
    )

    return {
        "experiment": "latent",
        "dataset": ds.summary(),
        "preprocess": prep_info,
        "model_selection": {"grid": grid, "selected": best},
        "variance_partition": partition,
        "cross_population_prediction": {
            "shared_private_model_r2": float(np.mean(cross["model_r2"])),
            "rrr_r2_by_rank": rrr_curve,
            "cca_heldout_canonical_corr": cca_mean,
        },
        "heldout_loglik_per_bin": {k: float(np.mean(v)) for k, v in fa_ll.items()},
        "decoding": {k: asdict(v) for k, v in dec.items()},
        "config": asdict(cfg) | {"preprocess": asdict(pcfg)},
    }


@dataclass(frozen=True)
class CouplingConfig:
    lam_grid: tuple[float, ...] = (0.0, 10.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)
    ridge: float = 1.0
    n_splits: int = 5
    n_shuffles: int = 20
    seed: int = 0


def run_coupling_experiment(
    s: np.ndarray,
    area_dims: list[int],
    conn: np.ndarray,
    blocks: np.ndarray,
    area_names: list[str],
    cfg: CouplingConfig | None = None,
) -> dict:
    """Experiment 2: does the atlas constrain the fitted inter-area coupling?"""
    cfg = cfg or CouplingConfig()
    rng = np.random.default_rng(cfg.seed)
    gkf = GroupKFold(n_splits=min(cfg.n_splits, np.unique(blocks).size))
    folds = list(gkf.split(s, groups=blocks))

    def cv_score(c: np.ndarray | None, lam: float) -> float:
        out = []
        for tr, te in folds:
            m = CoupledLDS(area_dims, conn=c, lam=lam, ridge=cfg.ridge).fit(s[tr], blocks[tr])
            out.append(m.score(s[te], blocks[te])["r2_overall"])
        return float(np.mean(out))

    unconstrained = cv_score(None, 0.0)
    by_lam = {}
    for lam in cfg.lam_grid:
        true_r2 = cv_score(conn, lam)
        shuf = [cv_score(shuffle_conn(conn, rng), lam) for _ in range(cfg.n_shuffles)]
        by_lam[str(lam)] = {
            "atlas_r2": true_r2,
            "shuffled_atlas_r2_mean": float(np.mean(shuf)),
            "shuffled_atlas_r2_p95": float(np.percentile(shuf, 95)),
            "advantage_over_shuffle": float(true_r2 - np.mean(shuf)),
        }
    best_lam = max(by_lam, key=lambda k: by_lam[k]["atlas_r2"])
    final = CoupledLDS(area_dims, conn=conn, lam=float(best_lam), ridge=cfg.ridge).fit(s, blocks)
    norms = final.block_norms()
    off = ~np.eye(len(area_dims), dtype=bool)
    corr = (
        float(np.corrcoef(norms[off], np.asarray(conn)[off])[0, 1])
        if off.sum() > 2
        else float("nan")
    )
    return {
        "experiment": "coupling",
        "areas": area_names,
        "area_dims": list(area_dims),
        "unconstrained_r2": unconstrained,
        "by_lambda": by_lam,
        "selected_lambda": float(best_lam),
        "coupling_block_norms": norms.tolist(),
        "connectivity": np.asarray(conn).tolist(),
        "corr_blocknorm_vs_connectivity_offdiag": corr,
        "config": asdict(cfg),
    }
