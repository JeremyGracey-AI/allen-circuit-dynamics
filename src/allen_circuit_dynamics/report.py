"""Figures for the two experiments. One PNG per question, nothing decorative."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_latent(metrics: dict, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    # 1. variance partition
    part = metrics["variance_partition"]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    names = list(part)
    bottoms = np.zeros(len(names))
    for comp, color in (("shared", "#534AB7"), ("private", "#1D9E75"), ("noise", "#B4B2A9")):
        vals = np.array([part[p][comp] for p in names])
        ax.bar(names, vals, bottom=bottoms, label=comp, color=color)
        bottoms += vals
    ax.set_ylabel("fraction of variance")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Shared / private / noise partition", fontsize=10)
    fig.tight_layout()
    p = out_dir / "variance_partition.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(str(p))

    # 2. model selection surface
    grid = metrics["model_selection"]["grid"]
    ds_vals = sorted({g["d_shared"] for g in grid})
    dp_vals = sorted({g["d_private"] for g in grid})
    z = np.full((len(dp_vals), len(ds_vals)), np.nan)
    for g in grid:
        z[dp_vals.index(g["d_private"]), ds_vals.index(g["d_shared"])] = g["heldout_loglik"]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    im = ax.imshow(z, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(ds_vals)), ds_vals)
    ax.set_yticks(range(len(dp_vals)), dp_vals)
    ax.set_xlabel("shared dims")
    ax.set_ylabel("private dims (each population)")
    sel = metrics["model_selection"]["selected"]
    ax.scatter(
        ds_vals.index(sel["d_shared"]),
        dp_vals.index(sel["d_private"]),
        marker="*",
        s=160,
        c="white",
        edgecolors="black",
    )
    fig.colorbar(im, ax=ax, label="held-out log-lik / bin")
    ax.set_title("Model selection (star = selected)", fontsize=10)
    fig.tight_layout()
    p = out_dir / "model_selection.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(str(p))

    # 3. cross-population prediction vs RRR rank
    cp = metrics["cross_population_prediction"]
    ranks = sorted(int(r) for r in cp["rrr_r2_by_rank"])
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(
        ranks,
        [
            cp["rrr_r2_by_rank"][str(r)]
            if str(r) in cp["rrr_r2_by_rank"]
            else cp["rrr_r2_by_rank"][r]
            for r in ranks
        ],
        "o-",
        label="reduced-rank regression",
    )
    ax.axhline(
        cp["shared_private_model_r2"],
        color="#D85A30",
        ls="--",
        label=f"shared/private model (d_shared={sel['d_shared']})",
    )
    ax.set_xlabel("rank of pop1 → pop2 map")
    ax.set_ylabel("held-out R² (pop2 from pop1)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("How many shared dimensions?", fontsize=10)
    fig.tight_layout()
    p = out_dir / "cross_population_prediction.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(str(p))

    # 4. decoding by subspace
    dec = metrics["decoding"]
    keys = [
        k
        for k in (
            "shared",
            "private1",
            "private2",
            "shared+private1",
            "all_latents",
            "raw_pop1",
            "raw_pop2",
            "raw_both",
        )
        if k in dec
    ]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    means = [dec[k]["r2_mean"] for k in keys]
    stds = [dec[k]["r2_std"] for k in keys]
    nulls = [dec[k]["null_r2_p95"] for k in keys]
    xs = np.arange(len(keys))
    ax.bar(xs, means, yerr=stds, color="#378ADD", capsize=3)
    ax.plot(xs, nulls, "k_", ms=14, label="shift-null 95th pct")
    ax.set_xticks(xs, keys, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("held-out R² (behavior)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Which subspace carries the behavior?", fontsize=10)
    fig.tight_layout()
    p = out_dir / "decoding_by_subspace.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(str(p))
    return written


def plot_coupling(metrics: dict, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    lams = list(metrics["by_lambda"])
    fig, ax = plt.subplots(figsize=(5, 3.2))
    xs = np.arange(len(lams))
    ax.plot(
        xs, [metrics["by_lambda"][k]["atlas_r2"] for k in lams], "o-", label="atlas-constrained"
    )
    ax.plot(
        xs,
        [metrics["by_lambda"][k]["shuffled_atlas_r2_mean"] for k in lams],
        "s--",
        label="shuffled atlas (mean)",
    )
    ax.axhline(metrics["unconstrained_r2"], color="gray", ls=":", label="unconstrained ridge")
    ax.set_xticks(xs, lams)
    ax.set_xlabel("λ (connectivity penalty)")
    ax.set_ylabel("held-out one-step R²")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Does the atlas constrain the dynamics?", fontsize=10)
    fig.tight_layout()
    p = out_dir / "coupling_vs_lambda.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(str(p))

    norms = np.array(metrics["coupling_block_norms"])
    conn = np.array(metrics["connectivity"])
    names = metrics["areas"]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    for ax, mat, title in (
        (axes[0], conn, "atlas projection density"),
        (axes[1], norms, "fitted coupling block norm"),
    ):
        im = ax.imshow(mat, cmap="magma")
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(names)), names, fontsize=7)
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    p = out_dir / "coupling_vs_connectivity.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(str(p))
    return written
