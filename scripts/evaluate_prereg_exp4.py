"""Score Experiment 4 against docs/experiment4_preregistration.md from the written metrics.
usage: .venv/bin/python scripts/evaluate_prereg_exp4.py outputs/lead_aind
Prints the per-session table and the pooled H5-H7 verdicts (re-derived from the per-session
files, so it does not trust the pooled metrics.json). Thresholds are the pre-registered ones.
"""

import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
rows = []
for mp in sorted(root.glob("*/metrics.json")):
    m = json.loads(mp.read_text())
    if m.get("experiment") != "lead":
        continue
    r = {"session": m["session"][8:], "genotype": (m.get("genotype") or "?")[:14]}
    for t in ("D1", "D2", "untagged"):
        bt = m["by_type"].get(t, {"n": 0, "n_coupled": 0, "coupled_peak_lags_s": []})
        r[t] = (bt["n"], bt["n_coupled"], bt["coupled_peak_lags_s"])
    r["pop"] = {}
    for name, p in m["population"].items():
        xc = p.get("mean_rate_xcorr_post_hoc", {})
        r["pop"][name] = (
            p["peak_lag_s"],
            p["peak_above_null"],
            xc.get("peak_lag_s", float("nan")),
            xc.get("coupled", False),
        )
    rows.append(r)

print(
    f"{'session':27s} {'genotype':14s}  D1 n/cpl med   D2 n/cpl med   untag n/cpl med   pop-xcorr D1 / D2 (post hoc)"
)


def cell(r, t):
    n, c, lags = r[t]
    med = f"{np.median(lags):+.2f}" if lags else "  —  "
    return f"{n:3d}/{c:3d} {med}"


for r in rows:
    px = " / ".join(
        f"{r['pop'][t][2]:+.2f}{'*' if r['pop'][t][3] else ' '}" if t in r["pop"] else "  —   "
        for t in ("D1", "D2")
    )
    print(
        f"{r['session']:27s} {r['genotype']:14s}  {cell(r, 'D1')}   {cell(r, 'D2')}   {cell(r, 'untagged')}   {px}"
    )

MIN_UNITS = 3
rng = np.random.default_rng(0)


def pooled(t):
    lags, n, c, pops = [], 0, 0, []
    for r in rows:
        nn, cc, ll = r[t]
        if nn < MIN_UNITS:
            continue
        n += nn
        c += cc
        lags.extend(ll)
        if t in r["pop"]:
            pops.append(r["pop"][t])
    arr = np.asarray(lags)
    meds = [np.median(rng.choice(arr, arr.size)) for _ in range(1000)] if arr.size else [np.nan]
    return {
        "n": n,
        "coupled": c,
        "frac": c / n if n else np.nan,
        "lags": arr,
        "median": float(np.median(arr)) if arr.size else np.nan,
        "ci": (float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))),
        "pop_neg_above": sum(1 for p in pops if p[0] < 0 and p[1]),
        "pop_n": len(pops),
        "xc_neg_coupled": sum(1 for p in pops if p[2] < 0 and p[3]),
    }


P = {t: pooled(t) for t in ("D1", "D2")}
print()
for t, p in P.items():
    lo, hi = p["ci"]
    print(
        f"{t}: {p['coupled']}/{p['n']} coupled ({p['frac']:.2f}); median lag {p['median']:+.3f} s "
        f"[{lo:+.3f}, {hi:+.3f}]; ridge population negative+above-null {p['pop_neg_above']}/{p['pop_n']}; "
        f"post hoc mean-rate xcorr negative+coupled {p['xc_neg_coupled']}/{p['pop_n']}"
    )
d1, d2 = P["D1"]["lags"], P["D2"]["lags"]
if d1.size and d2.size:
    diffs = [
        np.median(rng.choice(d1, d1.size)) - np.median(rng.choice(d2, d2.size)) for _ in range(1000)
    ]
    dlo, dhi = np.percentile(diffs, [2.5, 97.5])
    print(
        f"D1 - D2 median lag {np.median(d1) - np.median(d2):+.3f} s, CI95 [{dlo:+.3f}, {dhi:+.3f}]"
    )
else:
    dlo, dhi = np.nan, np.nan

print()
h5a = {t: (-0.5 <= P[t]["median"] <= -0.05) and P[t]["ci"][1] < 0 for t in P}
h5b = {t: P[t]["pop_n"] > 0 and P[t]["pop_neg_above"] >= np.ceil(2 * P[t]["pop_n"] / 3) for t in P}
h6 = bool(np.isfinite(dlo) and dlo >= -0.1 and dhi <= 0.1)
h7 = {t: P[t]["frac"] >= 0.30 for t in P}
for name, ok in (
    ("H5a pooled lead (median in [-0.5,-0.05] s, CI upper < 0, both types)", all(h5a.values())),
    (
        "H5b population ridge lead (negative + above null in >= 2/3 sessions, both)",
        all(h5b.values()),
    ),
    ("H6  same lead (CI95 of median difference within +-0.1 s)", h6),
    ("H7  coupled fraction >= 0.30, both types", all(h7.values())),
):
    print(f"{'SUPPORTED' if ok else 'NOT SUPPORTED':14s} {name}")
print(f"  per type: H5a {h5a}  H5b {h5b}  H7 {h7}")
