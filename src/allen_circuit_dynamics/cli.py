"""Command-line entry points.

acd demo    --out outputs/demo            # synthetic, both experiments, no data needed
acd fetch   --session 715093703 --areas VISp VISl --out data/ecephys   # needs allensdk
acd latent  --data data/ecephys/715093703_VISp_VISl.npz --out outputs/latent
acd coupling --data data/ecephys/715093703_areas.npz --conn data/conn.npy --out outputs/coupling
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
from .report import plot_coupling, plot_latent
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
        f"{m['corr_blocknorm_vs_connectivity_offdiag']:.3f}"
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="acd", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run both experiments on synthetic data")
    d.add_argument("--out", default="outputs/demo")
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(fn=cmd_demo)

    f = sub.add_parser("fetch", help="pull one Allen Neuropixels session (needs allensdk)")
    f.add_argument("--session", type=int, required=True)
    f.add_argument(
        "--areas", nargs="*", default=[],
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

    args = p.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
