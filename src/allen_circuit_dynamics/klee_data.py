"""Loader for the Klee et al. (2026) D1 / D2 striatal two-photon dataset.

Klee, Fernando-Peiris, Athalye, Costa, Peterka & Sippy (2026), *Science
Advances*, "Shared striatal neurons exhibit context-specific dynamics for
internally and externally driven actions." Data: Zenodo 10.5281/zenodo.16333867
(CC BY 4.0), file ``Data Structures/Switching_2p_Struc_Main.mat`` (MATLAB v7.3,
read with h5py). Six mice, one session each, anterior dorsal striatum at 30 fps.

Per session the file holds, per frame: neuropil-corrected fluorescence for
every ROI (``cells_raw_neurophil_corrected``, 0.7 scalar already applied),
lever displacement from a rotary encoder (``Joystick``, encoder units) and
video motion energy (``VidChangeZ``); per ROI a D1 / D2 label
(``D1D2class``: 1 = D1-SPN, 2 = D2-SPN, 0 = unclassified); and an event table
(``EventTS`` frame indices, ``BlockType`` 1 / 2) from which a per-frame block
label is forward-filled. Every session begins with ``BlockType`` 2 and the
paper says sessions begin with a cued block, so 2 is read as *cued* and 1 as
*self-paced*; that is an inference and is labelled as one in the metadata.

Preprocessing follows the paper where it states a recipe: ΔF/F against a
baseline that is the 50th percentile of the Gaussian-smoothed trace in
non-overlapping 60 s windows (interpolated between window centres), then a
z-score over the session. Frames are then averaged into bins.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dataset import PopulationDataset

FPS = 30.0
MOUSE_NAMES = ("JK238", "JK239", "JK244", "JK323", "JK324", "JK326")


@dataclass(frozen=True)
class KleeConfig:
    mat_path: Path
    mouse: int  # 0..5
    bin_frames: int = 3  # 100 ms at 30 fps
    baseline_window_s: float = 60.0
    smooth_sigma_frames: float = 3.0
    n_blocks: int = 12


def _read_string(f, ref) -> str:
    return "".join(chr(int(c)) for c in np.asarray(f[ref]).ravel())


def read_session(mat_path: Path, mouse: int) -> dict:
    """Raw per-session arrays, frames truncated to the common length."""
    import h5py

    with h5py.File(mat_path, "r") as f:
        g = f["Switching_2p_Struc_Main"]

        def get(name):
            return np.asarray(f[g[name][mouse, 0]])

        F = get("cells_raw_neurophil_corrected").astype(np.float64)  # (frames, cells)
        labels = get("D1D2class").ravel().astype(int)
        lever = get("Joystick").ravel().astype(np.float64)
        motion = get("VidChangeZ").ravel().astype(np.float64)
        ev_t = get("EventTS").ravel().astype(int)
        ev_block = get("BlockType").ravel().astype(int)
        name = _read_string(f, g["AnimalName"][mouse, 0])
        genotype = _read_string(f, g["Genotype"][mouse, 0])
        area = _read_string(f, g["ImagingArea2P"][mouse, 0])

    n = min(F.shape[0], lever.size, motion.size)
    dropped = {"frames_imaging": int(F.shape[0]), "frames_lever": int(lever.size), "frames_used": n}
    F, lever, motion = F[:n], lever[:n], motion[:n]

    order = np.argsort(ev_t, kind="stable")
    ev_t, ev_block = ev_t[order], ev_block[order]
    idx = np.searchsorted(ev_t, np.arange(n), side="right") - 1
    block_type = np.where(idx >= 0, ev_block[np.clip(idx, 0, None)], ev_block[0])
    return {
        "F": F,
        "labels": labels,
        "lever": lever,
        "motion": motion,
        "block_type": block_type,
        "name": name,
        "genotype": genotype,
        "area": area,
        "frames": dropped,
    }


def dff(
    F: np.ndarray, window_s: float = 60.0, sigma_frames: float = 3.0, fps: float = FPS
) -> np.ndarray:
    """ΔF/F with a running 50th-percentile baseline per cell (paper's recipe)."""
    from scipy.ndimage import gaussian_filter1d

    sm = gaussian_filter1d(F, sigma=sigma_frames, axis=0, mode="nearest")
    w = int(round(window_s * fps))
    n = F.shape[0]
    starts = np.arange(0, n, w)
    centres = np.minimum(starts + w / 2, n - 1)
    meds = np.stack([np.median(sm[s : s + w], axis=0) for s in starts])  # (n_win, cells)
    f0 = np.empty_like(F)
    t = np.arange(n)
    for c in range(F.shape[1]):
        f0[:, c] = np.interp(t, centres, meds[:, c])
    f0 = np.maximum(f0, 1e-6)
    return (F - f0) / f0


def zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std(axis=0)
    return (x - x.mean(axis=0)) / np.where(sd < 1e-8, 1.0, sd)


def bin_frames(x: np.ndarray, k: int, how: str = "mean") -> np.ndarray:
    n = (x.shape[0] // k) * k
    y = x[:n].reshape(n // k, k, *x.shape[1:])
    if how == "mean":
        return y.mean(axis=1)
    if how == "mode":
        # majority label per bin (small integer labels)
        out = np.empty(n // k, dtype=int)
        for i in range(n // k):
            v, cnt = np.unique(y[i], return_counts=True)
            out[i] = v[np.argmax(cnt)]
        return out
    raise ValueError(how)


def load_klee(cfg: KleeConfig) -> PopulationDataset:
    """D1 as population 1, D2 as population 2, lever displacement as behavior."""
    raw = read_session(cfg.mat_path, cfg.mouse)
    z = zscore(dff(raw["F"], cfg.baseline_window_s, cfg.smooth_sigma_frames))
    k = cfg.bin_frames
    x = bin_frames(z, k)
    lever = bin_frames(raw["lever"], k)
    motion = bin_frames(raw["motion"], k)
    block_type = bin_frames(raw["block_type"], k, how="mode")
    n_bins = x.shape[0]
    blocks = np.minimum(np.arange(n_bins) * cfg.n_blocks // n_bins, cfg.n_blocks - 1)
    d1 = raw["labels"] == 1
    d2 = raw["labels"] == 2
    context = np.where(block_type == 2, 1, 0)  # 1 = cued (inferred), 0 = self-paced
    meta = {
        "source": "Klee et al. 2026, Zenodo 10.5281/zenodo.16333867 (CC BY 4.0)",
        "mouse_index": cfg.mouse,
        "animal": raw["name"],
        "genotype": raw["genotype"],
        "area": raw["area"],
        "fps": FPS,
        "bin_frames": k,
        "frames": raw["frames"],
        "n_rois": {
            "D1": int(d1.sum()),
            "D2": int(d2.sum()),
            "unclassified": int((raw["labels"] == 0).sum()),
        },
        "behavior": "lever_displacement_encoder_units",
        "context_labels": {
            "1": (
                "cued (BlockType 2; inferred: every session starts with BlockType 2 and the "
                "paper says sessions start cued)"
            ),
            "0": "self-paced (BlockType 1)",
        },
        "preprocess": (
            "dF/F (60 s running 50th-percentile baseline of Gaussian-smoothed trace) -> z-score "
            "-> bin mean"
        ),
    }
    return PopulationDataset(
        x1=x[:, d1],
        x2=x[:, d2],
        behavior=lever,
        blocks=blocks,
        bin_s=k / FPS,
        pop_names=("D1", "D2"),
        meta=meta,
        extras={"context": context.astype(float), "motion": motion},
    )
