"""Post hoc checks for Experiment 4: resample animals, not neurons.

Not pre-registered; added 2026-11-01. The pre-registered H5a bootstrap resamples coupled
units as if independent. Units from one session share a mouse, a probe and a running
trace, so the interval that matters for a claim about striatum is the one that resamples
mice, then sessions within a mouse, then units within a session.

Also: tagged vs untagged coupled units within the same session (same speed trace, same
probe), the comparison behind the write-up's earlier, now withdrawn, statement that
tagged neurons lead by more than the untagged mix.

Eligibility as pre-registered: a session counts for a type if it has >= 3 QC-passing
units of that type; lags are the coupled units' peak lags.

usage: .venv/bin/python scripts/posthoc_exp4_hierarchical.py outputs/lead_aind [--boot 4000]
Writes outputs/lead_aind/posthoc_hierarchical.json and prints a summary.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = Path(args.root)
    rng = np.random.default_rng(args.seed)
    sessions = [
        json.loads(p.read_text())
        for p in sorted(root.glob("ecephys_*/metrics.json"))
        if json.loads(p.read_text()).get("experiment") == "lead"
    ]

    def units(m, t):
        return [u for u in m["units"] if u["type"] == t]

    out = {"note": "post hoc, not pre-registered; see docstring", "boot": args.boot, "types": {}}
    for t in ("D1", "D2"):
        by_mouse: dict[str, list[np.ndarray]] = {}
        for m in sessions:
            us = units(m, t)
            if len(us) < 3:
                continue
            lags = [u["peak_lag_s"] for u in us if u["coupled"]]
            if lags:
                by_mouse.setdefault(m["subject_id"], []).append(np.asarray(lags))
        mice = list(by_mouse)
        pooled = np.concatenate([x for v in by_mouse.values() for x in v])
        meds = []
        for _ in range(args.boot):
            draw = []
            for mi in rng.choice(len(mice), len(mice)):
                sess = by_mouse[mice[mi]]
                for si in rng.choice(len(sess), len(sess)):
                    draw.append(rng.choice(sess[si], sess[si].size))
            meds.append(np.median(np.concatenate(draw)))
        lo, hi = np.percentile(meds, [2.5, 97.5])
        per_mouse = {k: float(np.median(np.concatenate(v))) for k, v in by_mouse.items()}
        out["types"][t] = {
            "n_mice": len(mice),
            "n_sessions": sum(len(v) for v in by_mouse.values()),
            "n_coupled_units": int(pooled.size),
            "median_lag_s": float(np.median(pooled)),
            "hierarchical_ci95_s": [float(lo), float(hi)],
            "per_mouse_median_lag_s": per_mouse,
            "mice_with_negative_median": int(sum(v < 0 for v in per_mouse.values())),
        }
        print(
            f"{t}: {len(mice)} mice, {out['types'][t]['n_sessions']} sessions, "
            f"{pooled.size} coupled units | median {np.median(pooled):+.3f} s | "
            f"mouse>session>unit CI95 [{lo:+.3f}, {hi:+.3f}] | negative in "
            f"{out['types'][t]['mice_with_negative_median']}/{len(mice)} mice"
        )

    within = {}
    for t in ("D1", "D2"):
        diffs = []
        for m in sessions:
            a = [u["peak_lag_s"] for u in units(m, t) if u["coupled"]]
            b = [u["peak_lag_s"] for u in units(m, "untagged") if u["coupled"]]
            if len(a) >= 2 and len(b) >= 5:
                diffs.append(float(np.median(a) - np.median(b)))
        within[t] = {
            "n_sessions": len(diffs),
            "median_diff_s": float(np.median(diffs)) if diffs else float("nan"),
            "sessions_tagged_earlier": int(sum(d < 0 for d in diffs)),
            "per_session_diff_s": diffs,
        }
        print(
            f"{t} minus untagged, same session: {len(diffs)} sessions, median "
            f"{within[t]['median_diff_s']:+.3f} s, tagged earlier in "
            f"{within[t]['sessions_tagged_earlier']}/{len(diffs)}"
        )
    out["tagged_minus_untagged_within_session"] = within
    path = root / "posthoc_hierarchical.json"
    path.write_text(json.dumps(out, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
