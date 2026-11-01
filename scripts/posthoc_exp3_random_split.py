"""Post hoc control for Experiment 3: is the D1/D2 partition different from a random split?

Not pre-registered; added 2026-11-01 after the write-up called the equal shared fractions
"the robust finding". If cell-type identity carried population covariance of its own, the
D1/D2 split would show more private (population-specific) variance than random splits of
the same neurons into two groups of the same size. This script measures that.

Per mouse: the count-matched D1 and D2 ROIs of subsample draw 0, the (d_shared, d_private)
that held-out likelihood selected in Experiment 3 (mean over draws, rounded; d_private at
least 1 so a private subspace exists to measure), Experiment 3's preprocessing and model.
Then the same fit on --splits random two-way splits of the pooled ROIs.

usage: .venv/bin/python scripts/posthoc_exp3_random_split.py Switching_2p_Struc_Main.mat \
           outputs/force_klee [--splits 50] [--seed 0]
Writes outputs/force_klee/posthoc_random_split.json and prints one line per mouse.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from allen_circuit_dynamics.klee_data import MOUSE_NAMES, KleeConfig, load_klee
from allen_circuit_dynamics.models.shared_private import SharedPrivateFA
from allen_circuit_dynamics.preprocess import apply_scaler, fit_scaler


def partition(x1, x2, d_shared, d_private, seed=0):
    a, b = apply_scaler(x1, fit_scaler(x1)), apply_scaler(x2, fit_scaler(x2))
    vp = SharedPrivateFA(d_shared=d_shared, d_private=(d_private, d_private), seed=seed)
    vp = vp.fit(a, b).variance_partition()
    shared = (vp["pop1"]["shared"] + vp["pop2"]["shared"]) / 2
    private = (vp["pop1"]["private"] + vp["pop2"]["private"]) / 2
    return float(shared), float(private)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mat")
    ap.add_argument("metrics_root")
    ap.add_argument("--splits", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = Path(args.metrics_root)
    rng = np.random.default_rng(args.seed)
    out = {
        "note": "post hoc, not pre-registered; see docstring",
        "splits": args.splits,
        "seed": args.seed,
        "per_mouse": [],
    }
    for mi, name in enumerate(MOUSE_NAMES):
        t0 = time.time()
        m = json.loads((root / name / "metrics.json").read_text())
        sel = m["summary"]["all"]
        d_s = round(sel["selected_d_shared"]["mean"])
        d_p = max(1, round(sel["selected_d_private"]["mean"]))
        ds = load_klee(KleeConfig(mat_path=Path(args.mat), mouse=mi))
        draw = m["draws"][0]
        x1, x2 = ds.x1[:, draw["D1"]], ds.x2[:, draw["D2"]]
        true_shared, true_private = partition(x1, x2, d_s, d_p)
        pooled = np.hstack([x1, x2])
        n = x1.shape[1]
        null = []
        for _ in range(args.splits):
            perm = rng.permutation(pooled.shape[1])
            null.append(partition(pooled[:, perm[:n]], pooled[:, perm[n:]], d_s, d_p))
        null = np.asarray(null)
        p_private = (np.sum(null[:, 1] >= true_private) + 1) / (args.splits + 1)
        row = {
            "mouse": name,
            "n_per_population": int(n),
            "d_shared": d_s,
            "d_private": d_p,
            "d1_d2_shared": true_shared,
            "d1_d2_private": true_private,
            "random_shared_mean": float(null[:, 0].mean()),
            "random_private_mean": float(null[:, 1].mean()),
            "random_private_p95": float(np.percentile(null[:, 1], 95)),
            "p_private_ge_random": float(p_private),
        }
        out["per_mouse"].append(row)
        print(
            f"{name}: n={n} d=({d_s},{d_p}) | D1/D2 shared {true_shared:.3f} private "
            f"{true_private:.3f} | random splits shared {row['random_shared_mean']:.3f} "
            f"private {row['random_private_mean']:.3f} (p95 {row['random_private_p95']:.3f}) | "
            f"p={p_private:.3f} | {time.time() - t0:.0f}s",
            flush=True,
        )
    path = root / "posthoc_random_split.json"
    path.write_text(json.dumps(out, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
