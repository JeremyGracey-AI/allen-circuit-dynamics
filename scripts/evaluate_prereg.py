"""Score Experiment 3 against docs/experiment3_preregistration.md, mouse by mouse.
usage: .venv/bin/python scripts/evaluate_prereg.py outputs/force_klee  [--write]
Writes outputs/force_klee/prereg_evaluation.json (and prints a table). The thresholds
are the pre-registered ones; nothing here is tuned after seeing the data.
"""

import json
import sys
from pathlib import Path


def _shared_frac(summary):
    return {
        c: 0.5
        * (
            summary[c]["variance_partition"]["D1"]["shared"]["mean"]
            + summary[c]["variance_partition"]["D2"]["shared"]["mean"]
        )
        for c in summary
    }


def count(key):
    return sum(1 for r in rows if r.get(key))


root = Path(sys.argv[1])
rows = []
for mp in sorted(root.glob("*/metrics.json")):
    m = json.loads(mp.read_text())
    s = m["summary"]
    a = s["all"]
    d, p = a["decoding_r2"], a["decoding_null_p95"]
    bin_s = m["dataset"]["bin_s"]
    r = {
        "mouse": mp.parent.name,
        "n_matched": m["count_matched_n"],
        "d_shared": a["selected_d_shared"]["mean"],
        "d_private": a["selected_d_private"]["mean"],
        "shared_r2": d["shared"]["mean"],
        "shared_null_p95": p["shared"]["mean"],
        "shared_r2_motion_regressed": a["decoding_motion_regressed_r2"]["shared"]["mean"],
        "private_both_r2": d.get("private_both", {"mean": float("nan")})["mean"],
        "raw_D1_r2": d["raw_D1"]["mean"],
        "raw_D2_r2": d["raw_D2"]["mean"],
        "raw_both_r2": d["raw_both"]["mean"],
        "shared_frac_D1": a["variance_partition"]["D1"]["shared"]["mean"],
        "shared_frac_D2": a["variance_partition"]["D2"]["shared"]["mean"],
        "lag_peak_shared_s": a["lag_peak_bins"]["shared"]["mean"] * bin_s,
        "lag_peak_shared_sd_s": a["lag_peak_bins"]["shared"]["sd"] * bin_s,
    }
    # H1: |R²(D1) - R²(D2)| < 0.10
    r["H1_pass"] = abs(r["raw_D1_r2"] - r["raw_D2_r2"]) < 0.10
    # H2: shared > null p95 AND shared > private_both
    r["H2a_pass"] = r["shared_r2"] > r["shared_null_p95"]
    r["H2b_pass"] = (
        r["shared_r2"] > r["private_both_r2"]
        if r["private_both_r2"] == r["private_both_r2"]
        else None
    )
    # H3: shared lag peak negative, 100-700 ms lead
    lag = r["lag_peak_shared_s"]
    r["H3_pass"] = -0.7 <= lag <= -0.1
    # H4: shared fraction differs < 0.05 between contexts; lead differs > 0.1 s
    if "cued" in s and "selfpaced" in s:
        sf = _shared_frac(s)
        r["shared_frac_cued"], r["shared_frac_selfpaced"] = sf["cued"], sf["selfpaced"]
        r["H4a_pass"] = abs(r["shared_frac_cued"] - r["shared_frac_selfpaced"]) < 0.05
        lc = s["cued"]["lag_peak_bins"]["shared"]["mean"] * bin_s
        ls = s["selfpaced"]["lag_peak_bins"]["shared"]["mean"] * bin_s
        r["lag_cued_s"], r["lag_selfpaced_s"] = lc, ls
        r["H4b_pass"] = abs(lc - ls) > 0.1
    rows.append(r)

n = len(rows)
verdict = {
    "H1 (D1 ~ D2 decode, |diff|<0.10 in >=5/6)": (count("H1_pass"), 5),
    "H2a (shared > shift-null p95 in >=5/6)": (count("H2a_pass"), 5),
    "H2b (shared > private_both in >=4/6)": (count("H2b_pass"), 4),
    "H3 (shared lag peak -0.7..-0.1 s in >=4/6)": (count("H3_pass"), 4),
    "H4a (shared fraction |cued-selfpaced|<0.05 in most)": (count("H4a_pass"), (n // 2) + 1),
    "H4b (lead differs >0.1 s between contexts in most)": (count("H4b_pass"), (n // 2) + 1),
}
HYP = ("H1_pass", "H2a_pass", "H2b_pass", "H3_pass", "H4a_pass", "H4b_pass")
HEAD = [
    "mouse",
    "n",
    "dS",
    "dP",
    "shared",
    "null95",
    "motion",
    "privB",
    "rawD1",
    "rawD2",
    "both",
    "lag(s)",
]
print(" ".join(f"{h:>7s}" for h in HEAD) + "  H1 H2a H2b H3 H4a H4b")
for r in rows:
    print(
        f"{r['mouse']:7s} {r['n_matched']:4d} {r['d_shared']:4.1f} {r['d_private']:4.1f} "
        f"{r['shared_r2']:7.3f} {r['shared_null_p95']:7.3f} {r['shared_r2_motion_regressed']:7.3f} "
        f"{r['private_both_r2']:7.3f} {r['raw_D1_r2']:7.3f} {r['raw_D2_r2']:7.3f} "
        f"{r['raw_both_r2']:7.3f} "
        f"{r['lag_peak_shared_s']:+7.2f} "
        + " ".join(
            "Y" if r.get(k) else "n"
            for k in ("H1_pass", "H2a_pass", "H2b_pass", "H3_pass", "H4a_pass", "H4b_pass")
        )
    )
print()
out = {"n_mice": n, "per_mouse": rows, "verdicts": {}}
for k, (got, need) in verdict.items():
    status = "SUPPORTED" if got >= need else "NOT SUPPORTED"
    out["verdicts"][k] = {"passing_mice": got, "required": need, "status": status}
    print(f"{status:14s} {got}/{n} (needed {need})  {k}")
if "--write" in sys.argv:
    (root / "prereg_evaluation.json").write_text(json.dumps(out, indent=2))
    print("wrote", root / "prereg_evaluation.json")
