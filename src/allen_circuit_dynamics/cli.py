"""Command-line entry points.

acd demo     --out outputs/demo                       # synthetic, experiments 1-2, no data needed
acd fetch    --session 715093703 --out data/ecephys    # needs the allen extra (allensdk)
acd latent   --data data/ecephys/<session>_<a1>_<a2>.npz --out outputs/latent
acd coupling --data data/ecephys/<s>_areas.npz --conn data/conn_<s>.npy --out outputs/coupling
acd force    --mat Switching_2p_Struc_Main.mat --mouse all --out outputs/force_klee   # experiment 3
acd lead     --sessions all --out outputs/lead_aind     # experiment 4 (needs the aind extra)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .dataset import PopulationDataset
from .experiments import (
    CouplingConfig,
    LatentConfig,
    run_coupling_experiment,
    run_latent_experiment,
)
from .provenance import provenance
from .report import plot_coupling, plot_force, plot_latent, plot_lead, plot_lead_pooled
from .synth import CoupledSynthConfig, SynthConfig, make_synthetic, make_synthetic_coupled


def _write_metrics(metrics: dict, out: Path, config: object, data_meta: dict) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    metrics = {"provenance": provenance(config, data_meta), **metrics}
    path = out / "metrics.json"
    with open(path, "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    return path


def _summarise_latent(m: dict) -> str:
    sel = m["model_selection"]["selected"]
    vp = m["variance_partition"]
    dec = m["decoding"]
    lines = [
        f"selected: d_shared={sel['d_shared']} d_private={sel['d_private']} "
        f"(held-out loglik/bin {sel['heldout_loglik']:.3f})",
    ]
    for pop, v in vp.items():
        lines.append(
            f"{pop}: shared {v['shared']:.2f} / private {v['private']:.2f} / noise {v['noise']:.2f}"
        )
    cp = m["cross_population_prediction"]
    lines.append(
        f"pop2 from pop1: model R² {cp['shared_private_model_r2']:.3f}; "
        f"RRR by rank {', '.join(f'{k}:{v:.2f}' for k, v in cp['rrr_r2_by_rank'].items())}"
    )
    for k in ("shared", "private1", "private2", "all_latents", "raw_both"):
        if k in dec:
            d = dec[k]
            lines.append(
                f"decode behavior from {k:<14} R² {d['r2_mean']:.3f} ± {d['r2_std']:.3f} "
                f"(shift-null p95 {d['null_r2_p95']:.3f})"
            )
    return "\n".join(lines)


def _summarise_coupling(m: dict) -> str:
    lines = [f"unconstrained ridge R² {m['unconstrained_r2']:.3f}"]
    for lam, v in m["by_lambda"].items():
        lines.append(
            f"λ={lam:>7}: atlas {v['atlas_r2']:.3f}  shuffled {v['shuffled_atlas_r2_mean']:.3f} "
            f"(p95 {v['shuffled_atlas_r2_p95']:.3f})  Δ {v['advantage_over_shuffle']:+.3f}"
        )
    lines.append(
        f"corr(block norm, connectivity) off-diag = "
        f"{m['corr_blocknorm_vs_connectivity_offdiag']:.3f} "
        f"(permutation p = {m['corr_permutation_p']:.3f}, null p95 {m['corr_null_p95']:.3f}); "
        f"within-area share of ||A||² = {m['within_area_fraction_of_coupling']:.2f}"
    )
    return "\n".join(lines)


# ----------------------------------------------------------------- commands
def cmd_demo(args: argparse.Namespace) -> int:
    out = Path(args.out)
    ds, _ = make_synthetic(SynthConfig(seed=args.seed))
    lcfg = LatentConfig(seed=args.seed)
    m1 = run_latent_experiment(ds, lcfg)
    p1 = _write_metrics(m1, out / "latent", lcfg, ds.meta)
    figs1 = plot_latent(m1, out / "latent")
    print("== Experiment 1: shared vs private (synthetic) ==")
    print(_summarise_latent(m1))
    print(f"wrote {p1} and {len(figs1)} figures")

    syn = make_synthetic_coupled(CoupledSynthConfig(seed=args.seed))
    ccfg = CouplingConfig(seed=args.seed)
    m2 = run_coupling_experiment(
        syn["s"], syn["area_dims"], syn["conn"], syn["blocks"], syn["area_names"], ccfg
    )
    p2 = _write_metrics(m2, out / "coupling", ccfg, {"source": "synthetic_coupled"})
    figs2 = plot_coupling(m2, out / "coupling")
    print("\n== Experiment 2: connectivity-constrained coupling (synthetic) ==")
    print(_summarise_coupling(m2))
    print(f"wrote {p2} and {len(figs2)} figures")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    from .allen_data import FetchConfig, fetch_session  # lazy: needs allensdk

    cfg = FetchConfig(
        session_id=args.session,
        areas=tuple(args.areas),
        min_units=args.min_units,
        max_areas=args.max_areas,
        bin_s=args.bin,
        cache_dir=Path(args.cache),
        stimulus=args.stimulus,
    )
    ds, areas_npz = fetch_session(cfg, Path(args.out))
    print(json.dumps(ds.summary(), indent=2))
    print(f"two-population dataset: {ds.meta.get('saved_to')}")
    print(f"all-area latents for coupling: {areas_npz}")
    return 0


def cmd_latent(args: argparse.Namespace) -> int:
    ds = PopulationDataset.load(args.data)
    cfg = LatentConfig(seed=args.seed)
    m = run_latent_experiment(ds, cfg)
    p = _write_metrics(m, Path(args.out), cfg, ds.meta)
    figs = plot_latent(m, Path(args.out))
    print(_summarise_latent(m))
    print(f"wrote {p} and {len(figs)} figures")
    return 0


def cmd_coupling(args: argparse.Namespace) -> int:
    with np.load(args.data, allow_pickle=False) as z:
        s, blocks = z["s"], z["blocks"]
        area_dims = [int(v) for v in z["area_dims"]]
        area_names = [str(v) for v in z["area_names"]]
    conn = np.load(args.conn)
    cfg = CouplingConfig(seed=args.seed)
    m = run_coupling_experiment(s, area_dims, conn, blocks, area_names, cfg)
    p = _write_metrics(m, Path(args.out), cfg, {"data": str(args.data), "conn": str(args.conn)})
    figs = plot_coupling(m, Path(args.out))
    print(_summarise_coupling(m))
    print(f"wrote {p} and {len(figs)} figures")
    return 0


def _summarise_force(m: dict) -> str:
    lines = []
    for c, s in m["summary"].items():
        d = s["decoding_r2"]
        p = s["decoding_null_p95"]
        mr = s["decoding_motion_regressed_r2"]
        lag = s["lag_peak_bins"]
        vp = s["variance_partition"]
        lines.append(
            f"[{c}] d_shared {s['selected_d_shared']['mean']:.1f} d_private "
            f"{s['selected_d_private']['mean']:.1f} | D1 shared/private/indep "
            f"{vp['D1']['shared']['mean']:.2f}/{vp['D1']['private']['mean']:.2f}/"
            f"{vp['D1']['independent']['mean']:.2f}, D2 {vp['D2']['shared']['mean']:.2f}/"
            f"{vp['D2']['private']['mean']:.2f}/{vp['D2']['independent']['mean']:.2f}"
        )
        lines.append(
            f"      lever R²: shared {d['shared']['mean']:.3f}±{d['shared']['sd']:.3f} "
            f"(null p95 {p['shared']['mean']:.3f}; motion-regressed {mr['shared']['mean']:.3f}) | "
            f"private_both {d['private_both']['mean']:.3f} | raw D1 {d['raw_D1']['mean']:.3f} "
            f"raw D2 {d['raw_D2']['mean']:.3f} raw both {d['raw_both']['mean']:.3f}"
        )
        lines.append(
            "      lag of peak (bins, <0 = neural leads): "
            + ", ".join(f"{k} {v['mean']:+.1f}±{v['sd']:.1f}" for k, v in lag.items())
        )
    return "\n".join(lines)


def cmd_force(args: argparse.Namespace) -> int:
    from .experiments_force import ForceConfig, run_force_experiment

    cfg = ForceConfig(seed=args.seed, n_draws=args.draws)
    out = Path(args.out)
    if args.data:
        datasets = [(Path(args.data).stem, PopulationDataset.load(args.data))]
    else:
        from .klee_data import KleeConfig, load_klee

        mice = range(6) if args.mouse == "all" else [int(args.mouse)]
        datasets = []
        for mi in mice:
            ds = load_klee(KleeConfig(mat_path=Path(args.mat), mouse=mi))
            datasets.append((f"{ds.meta['animal']}", ds))
    for name, ds in datasets:
        m = run_force_experiment(ds, cfg)
        sub = out / name
        p = _write_metrics(m, sub, cfg, ds.meta)
        figs = plot_force(m, sub)
        print(f"== {name}: D1 {ds.n_units[0]} D2 {ds.n_units[1]} units, {ds.n_bins} bins ==")
        print(_summarise_force(m))
        print(f"wrote {p} and {len(figs)} figures")
    return 0


def _summarise_lead(m: dict) -> str:
    lines = []
    for t, bt in m["by_type"].items():
        med = np.median(bt["coupled_peak_lags_s"]) if bt["coupled_peak_lags_s"] else float("nan")
        lines.append(
            f"{t:9s} n={bt['n']:3d} coupled={bt['n_coupled']:3d} median peak lag {med:+.3f} s"
        )
    for name, p in m["population"].items():
        flag = "above null" if p["peak_above_null"] else "below null"
        xc = p.get("mean_rate_xcorr_post_hoc")
        post = (
            f"; mean-rate xcorr peak {xc['peak_lag_s']:+.2f} s r {xc['peak_r']:+.2f}"
            f" ({'coupled' if xc['coupled'] else 'not coupled'}, post hoc)"
            if xc
            else ""
        )
        lines.append(
            f"pop {name:13s} n={p['n_units']:3d} ridge peak {p['peak_lag_s']:+.2f} s "
            f"R2 {p['peak_r2']:.3f} {flag}{post}"
        )
    fa = m.get("exploratory_fa")
    if fa:
        s = fa["selected"]
        vp = fa["variance_partition"]
        lines.append(
            f"exploratory FA ({fa['bin_s'] * 1000:.0f} ms): "
            f"d_shared={s['d_shared']} d_private={s['d_private']} "
            f"shared frac D1 {vp['D1']['shared']:.2f} D2 {vp['D2']['shared']:.2f}; "
            f"shared-latent peak lag {fa['shared_lag_profile']['peak_lag_s']:+.2f} s"
        )
    return "\n".join(lines)


def cmd_lead(args: argparse.Namespace) -> int:
    from .experiments_lead import LeadConfig, pool_lead, run_lead_session

    cfg = LeadConfig(seed=args.seed)
    out = Path(args.out)
    results: list[dict] = []
    skipped: dict[str, str] = {}

    def analyse(name, sess) -> None:
        m = run_lead_session(sess, cfg)
        _write_metrics(m, out / name, cfg, sess.meta)
        plot_lead(m, out / name)
        print(f"== {name}: {sess.types()} {sess.n_bins} bins ==")
        print(_summarise_lead(m))
        results.append(m)

    if args.synthetic:
        from .synth import LeadSynthConfig, make_synthetic_lead

        sess, _ = make_synthetic_lead(LeadSynthConfig(seed=args.seed))
        analyse("synthetic", sess)
    else:
        from .aind_opto import SESSIONS, AindOptoConfig, load_with_retries

        names = list(SESSIONS) if args.sessions == "all" else args.sessions.split(",")
        # One session at a time: fetch, analyse, write. A crash or kill loses at most one
        # session, and --resume picks up from the files already written.
        for name in names:
            done = out / name / "metrics.json"
            if args.resume and done.exists():
                with open(done) as fh:
                    results.append(json.load(fh))
                print(f"== {name}: cached")
                continue
            try:
                sess = load_with_retries(AindOptoConfig(session=name), cache_dir=args.cache)
            except Exception as e:  # one bad session must not stop the sweep
                skipped[name] = f"{type(e).__name__}: {e}"
                print(f"== {name}: skipped ({skipped[name]})")
                continue
            analyse(name, sess)
    if len(results) > 1 or not args.synthetic:
        pooled = pool_lead(results, cfg)
        pooled["skipped_sessions"] = skipped
        p = _write_metrics(pooled, out, cfg, {"sessions": [r["session"] for r in results]})
        plot_lead_pooled(pooled, out)
        v = pooled["verdicts"]
        print("== pooled ==")
        for t in ("D1", "D2"):
            pt = pooled["per_type"][t]
            lo, hi = pt["median_ci95_s"]
            print(
                f"{t}: {pt['n_coupled']}/{pt['n_units']} coupled over {pt['n_sessions']} sessions; "
                f"median lag {pt['median_lag_s']:+.3f} s [{lo:+.3f}, {hi:+.3f}]; "
                f"population negative+above-null "
                f"{pt['population_negative_above_null']}/{pt['population_profiles']}"
            )
        d = pooled["D1_minus_D2"]
        lo, hi = d["ci95_s"]
        print(f"D1-D2 median lag {d['median_diff_s']:+.3f} s CI95 [{lo:+.3f}, {hi:+.3f}]")
        print(
            f"H5a {v['H5a_pooled_lead']}  H5b {v['H5b_population_lead']}  "
            f"H6 {v['H6_same_lead']}  H7 {v['H7_coupled_fraction']}"
        )
        if skipped:
            print(f"skipped {len(skipped)} sessions (reasons in the pooled metrics.json)")
        print(f"wrote {p}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="acd", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    from . import __version__

    # Line-buffer stdout so `nohup acd ... > run.log` can be followed with `tail -f`.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run both experiments on synthetic data")
    d.add_argument("--out", default="outputs/demo")
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(fn=cmd_demo)

    f = sub.add_parser("fetch", help="pull one Allen Neuropixels session (needs allensdk)")
    f.add_argument("--session", type=int, required=True)
    f.add_argument(
        "--areas",
        nargs="*",
        default=[],
        help="visual areas to use; default: top areas by QC-passing unit count",
    )
    f.add_argument("--min-units", type=int, default=15, help="min QC-passing units per area")
    f.add_argument("--max-areas", type=int, default=6, help="max areas when auto-selecting")
    f.add_argument("--bin", type=float, default=0.05)
    f.add_argument("--stimulus", default="natural_movie_one")
    f.add_argument("--cache", default="data/ecephys_cache")
    f.add_argument("--out", default="data/ecephys")
    f.set_defaults(fn=cmd_fetch)

    l_ = sub.add_parser("latent", help="experiment 1 on a saved dataset")
    l_.add_argument("--data", required=True)
    l_.add_argument("--out", default="outputs/latent")
    l_.add_argument("--seed", type=int, default=0)
    l_.set_defaults(fn=cmd_latent)

    c = sub.add_parser("coupling", help="experiment 2 on saved area latents + connectivity matrix")
    c.add_argument("--data", required=True)
    c.add_argument("--conn", required=True)
    c.add_argument("--out", default="outputs/coupling")
    c.add_argument("--seed", type=int, default=0)
    c.set_defaults(fn=cmd_coupling)

    fo = sub.add_parser("force", help="experiment 3: D1/D2 shared-private structure and the lever")
    fo.add_argument("--mat", help="Klee et al. 2026 Switching_2p_Struc_Main.mat")
    fo.add_argument("--mouse", default="all", help="0-5 or 'all'")
    fo.add_argument("--data", help="instead of --mat: a saved PopulationDataset .npz with extras")
    fo.add_argument("--out", required=True)
    fo.add_argument("--draws", type=int, default=10)
    fo.add_argument("--seed", type=int, default=0)
    fo.set_defaults(fn=cmd_force)

    le = sub.add_parser(
        "lead", help="experiment 4: do tagged D1/D2 spikes lead running speed? (aind extra)"
    )
    le.add_argument(
        "--sessions", default="all", help="'all' or comma-separated ecephys_<subject>_<date>_<time>"
    )
    le.add_argument(
        "--cache", default="data/aind_opto", help="fetched sessions cached here as .npz"
    )
    le.add_argument("--out", required=True)
    le.add_argument("--resume", action="store_true", help="skip sessions whose metrics.json exists")
    le.add_argument("--synthetic", action="store_true", help="run on a planted synthetic session")
    le.add_argument("--seed", type=int, default=0)
    le.set_defaults(fn=cmd_lead)

    args = p.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
