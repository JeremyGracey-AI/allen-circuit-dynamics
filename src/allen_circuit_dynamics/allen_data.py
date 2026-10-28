"""Allen Brain Observatory — Visual Coding Neuropixels loader.

Pulls one session through ``allensdk``'s ``EcephysProjectCache`` (public S3
bucket, no credentials), bins spikes per unit, aligns running speed to the
same bins, and writes:

* a two-population :class:`PopulationDataset` (the first two ``areas``) for
  Experiment 1, with running speed as the continuous behavioral variable;
* an ``*_areas.npz`` with per-area PCA latents for *all* requested areas, for
  Experiment 2.

``allensdk`` is imported lazily so the rest of the package installs and tests
without it. Install with ``pip install -e ".[allen]"`` on the machine that
will hold the NWB files (1–3 GB per session).

Unit quality filters follow the Allen defaults that most analyses use:
``isi_violations < 0.5``, ``amplitude_cutoff < 0.1``, ``presence_ratio > 0.9``.
Blocks for cross-validation are contiguous chunks of the chosen stimulus
epoch, so folds never share a window of time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .dataset import PopulationDataset
from .preprocess import PreprocessConfig, apply_scaler, filter_units, fit_scaler, transform_counts


@dataclass(frozen=True)
class FetchConfig:
    session_id: int
    areas: tuple[str, ...] = ("VISp", "VISl")
    bin_s: float = 0.05
    stimulus: str = "natural_movie_one"
    cache_dir: Path = Path("data/ecephys_cache")
    n_blocks: int = 12
    isi_violations_max: float = 0.5
    amplitude_cutoff_max: float = 0.1
    presence_ratio_min: float = 0.9
    area_latent_dims: int = 5


def _epoch_windows(stim_table, stimulus: str) -> list[tuple[float, float]]:
    """Contiguous presentation windows for the named stimulus (merged)."""
    t = stim_table[stim_table["stimulus_name"] == stimulus].sort_values("start_time")
    if t.empty:
        raise ValueError(f"No presentations of stimulus '{stimulus}' in this session.")
    windows: list[list[float]] = []
    for s, e in zip(t["start_time"].to_numpy(), t["stop_time"].to_numpy(), strict=True):
        if windows and s - windows[-1][1] < 1.0:
            windows[-1][1] = e
        else:
            windows.append([float(s), float(e)])
    return [(a, b) for a, b in windows]


def _bin_windows(windows: list[tuple[float, float]], bin_s: float) -> list[np.ndarray]:
    """Bin edges per window; a bin never straddles two windows."""
    out = []
    for a, b in windows:
        e = np.arange(a, b, bin_s)
        if e.size >= 2:
            out.append(e)
    if not out:
        raise ValueError("Stimulus windows are shorter than one bin.")
    return out


def _bin_spikes(
    spike_times: dict[int, np.ndarray], unit_ids: np.ndarray, window_edges: list[np.ndarray]
) -> np.ndarray:
    """Spike counts, shape (n_bins_total, n_units), windows concatenated in time order."""
    cols = []
    for uid in unit_ids:
        st = np.asarray(spike_times[uid])
        cols.append(np.concatenate([np.histogram(st, bins=e)[0] for e in window_edges]))
    return np.stack(cols, axis=1).astype(float)


def _bin_speed(running_speed, window_edges: list[np.ndarray]) -> np.ndarray:
    """Mean running speed per bin; NaN where the speed trace has no samples."""
    t = 0.5 * (running_speed["start_time"].to_numpy() + running_speed["end_time"].to_numpy())
    v = running_speed["velocity"].to_numpy()
    out = []
    for e in window_edges:
        idx = np.searchsorted(e, t, side="right") - 1
        ok = (idx >= 0) & (idx < e.size - 1)
        ssum = np.zeros(e.size - 1)
        n = np.zeros(e.size - 1)
        np.add.at(ssum, idx[ok], v[ok])
        np.add.at(n, idx[ok], 1)
        out.append(np.where(n > 0, ssum / np.maximum(n, 1), np.nan))
    return np.concatenate(out)


def fetch_session(cfg: FetchConfig, out_dir: Path) -> tuple[PopulationDataset, Path]:
    from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

    cache = EcephysProjectCache.from_warehouse(manifest=str(cfg.cache_dir / "manifest.json"))
    session = cache.get_session_data(cfg.session_id)
    units = session.units
    good = units[
        (units["isi_violations"] < cfg.isi_violations_max)
        & (units["amplitude_cutoff"] < cfg.amplitude_cutoff_max)
        & (units["presence_ratio"] > cfg.presence_ratio_min)
    ]
    windows = _epoch_windows(session.stimulus_presentations, cfg.stimulus)
    window_edges = _bin_windows(windows, cfg.bin_s)
    speed = _bin_speed(session.running_speed, window_edges)
    keep_all = np.isfinite(speed)

    per_area: dict[str, np.ndarray] = {}
    for area in cfg.areas:
        uids = good[good["ecephys_structure_acronym"] == area].index.to_numpy()
        if uids.size < 5:
            raise ValueError(
                f"Area {area}: only {uids.size} units pass QC in session {cfg.session_id}."
            )
        per_area[area] = _bin_spikes(session.spike_times, uids, window_edges)

    # blocks: contiguous chunks across the kept bins
    kept = np.where(keep_all)[0]
    blocks = np.floor(np.arange(kept.size) * cfg.n_blocks / kept.size).astype(int)
    speed_k = speed[kept]
    meta_common = {
        "source": "allen_visual_coding_neuropixels",
        "session_id": cfg.session_id,
        "stimulus": cfg.stimulus,
        "n_windows": len(windows),
        "unit_qc": {
            "isi_violations_max": cfg.isi_violations_max,
            "amplitude_cutoff_max": cfg.amplitude_cutoff_max,
            "presence_ratio_min": cfg.presence_ratio_min,
        },
        "fetch_config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
        "n_bins_dropped_missing_speed": int((~keep_all).sum()),
    }

    a1, a2 = cfg.areas[0], cfg.areas[1]
    ds = PopulationDataset(
        x1=per_area[a1][kept],
        x2=per_area[a2][kept],
        behavior=speed_k,
        blocks=blocks,
        bin_s=cfg.bin_s,
        pop_names=(a1, a2),
        meta=meta_common
        | {
            "behavior": "running_speed_cm_s",
            "units_per_area": {a1: int(per_area[a1].shape[1]), a2: int(per_area[a2].shape[1])},
        },
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{cfg.session_id}_{a1}_{a2}"
    saved = ds.save(stem)
    ds.meta["saved_to"] = str(saved)
    with open(stem.with_suffix(".meta.json"), "w") as fh:
        import json

        json.dump(ds.meta, fh, indent=2, default=str)

    # per-area PCA latents for experiment 2
    pcfg = PreprocessConfig()
    lat, dims, names = [], [], []
    for area, counts in per_area.items():
        x, _ = filter_units(counts[kept], cfg.bin_s, pcfg.min_rate_hz)
        x, _ = transform_counts(x, pcfg)
        x = apply_scaler(x, fit_scaler(x))
        u, s, vt = np.linalg.svd(x - x.mean(0), full_matrices=False)
        d = min(cfg.area_latent_dims, vt.shape[0])
        lat.append((x - x.mean(0)) @ vt[:d].T)
        dims.append(d)
        names.append(area)
    areas_npz = out_dir / f"{cfg.session_id}_areas.npz"
    np.savez_compressed(
        areas_npz,
        s=np.hstack(lat),
        blocks=blocks,
        area_dims=np.array(dims),
        area_names=np.array(names),
    )
    return ds, areas_npz


def projection_matrix(areas: tuple[str, ...], cache_dir: Path) -> np.ndarray:
    """Row-normalised ipsilateral projection density between cortical areas from the
    Allen Mouse Connectivity Atlas (needs allensdk; downloads on first call)."""
    from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

    mcc = MouseConnectivityCache(manifest_file=str(cache_dir / "connectivity_manifest.json"))
    tree = mcc.get_structure_tree()
    ids = [tree.get_structures_by_acronym([a])[0]["id"] for a in areas]
    exps = mcc.get_experiments(dataframe=True, injection_structure_ids=ids)
    conn = np.eye(len(areas))
    for i, src in enumerate(ids):
        rows = exps[exps["structure_id"] == src]
        if rows.empty:
            continue
        pm = mcc.get_projection_matrix(
            experiment_ids=rows.index.to_list(),
            projection_structure_ids=ids,
            hemisphere_ids=[2],
            parameter="projection_density",
        )
        dens = np.nanmean(pm["matrix"], axis=0)
        for j in range(len(ids)):
            if i != j:
                conn[i, j] = float(dens[j])
    off = ~np.eye(len(areas), dtype=bool)
    if conn[off].max() > 0:
        conn[off] = conn[off] / conn[off].max()
    return np.clip(conn, 0, 1)
