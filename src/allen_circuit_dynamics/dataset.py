"""The one data container every stage reads and writes.

A :class:`PopulationDataset` is two binned population matrices recorded at the
same time bins, a continuous behavioral signal on those bins, and a block id
per bin so cross-validation can keep contiguous time together.

Shapes are validated on construction. Nothing downstream re-checks them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PopulationDataset:
    """Two populations, one behavioral signal, aligned on time bins.

    Attributes
    ----------
    x1, x2:
        ``(n_bins, n_units_i)`` binned activity (spike counts or rates).
    behavior:
        ``(n_bins,)`` continuous behavioral variable (e.g. running speed, force).
    blocks:
        ``(n_bins,)`` integer block id; bins in one block are contiguous in time
        and are never split across cross-validation folds.
    bin_s:
        Bin width in seconds.
    pop_names:
        Human-readable names for the two populations (e.g. brain areas).
    meta:
        Free-form provenance: session id, unit filters, source, versions.
    extras:
        Optional per-bin covariates, each ``(n_bins,)`` (e.g. a context label,
        a motion-energy trace). Saved alongside the populations.
    """

    x1: np.ndarray
    x2: np.ndarray
    behavior: np.ndarray
    blocks: np.ndarray
    bin_s: float
    pop_names: tuple[str, str] = ("pop1", "pop2")
    meta: dict = field(default_factory=dict)
    extras: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = self.x1.shape[0]
        if self.x1.ndim != 2 or self.x2.ndim != 2:
            raise ValueError("x1 and x2 must be 2-D (n_bins, n_units).")
        if self.x2.shape[0] != n:
            raise ValueError(f"x1 has {n} bins, x2 has {self.x2.shape[0]}.")
        if self.behavior.shape != (n,):
            raise ValueError(f"behavior must have shape ({n},), got {self.behavior.shape}.")
        if self.blocks.shape != (n,):
            raise ValueError(f"blocks must have shape ({n},), got {self.blocks.shape}.")
        if self.bin_s <= 0:
            raise ValueError("bin_s must be positive.")
        if not np.all(np.isfinite(self.x1)) or not np.all(np.isfinite(self.x2)):
            raise ValueError("Population matrices contain non-finite values.")
        if not np.all(np.isfinite(self.behavior)):
            raise ValueError("behavior contains non-finite values.")
        for k, v in self.extras.items():
            if v.shape != (n,):
                raise ValueError(f"extras[{k!r}] must have shape ({n},), got {v.shape}.")

    @property
    def n_bins(self) -> int:
        return int(self.x1.shape[0])

    @property
    def n_units(self) -> tuple[int, int]:
        return int(self.x1.shape[1]), int(self.x2.shape[1])

    @property
    def n_blocks(self) -> int:
        return int(np.unique(self.blocks).size)

    def summary(self) -> dict:
        return {
            "n_bins": self.n_bins,
            "n_units": list(self.n_units),
            "n_blocks": self.n_blocks,
            "bin_s": self.bin_s,
            "duration_s": round(self.n_bins * self.bin_s, 3),
            "pop_names": list(self.pop_names),
            "behavior_mean": float(np.mean(self.behavior)),
            "behavior_std": float(np.std(self.behavior)),
        }

    def save(self, path: Path | str) -> Path:
        """Write to ``<path>.npz`` plus ``<path>.meta.json``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path.with_suffix(".npz"),
            x1=self.x1,
            x2=self.x2,
            behavior=self.behavior,
            blocks=self.blocks,
            bin_s=np.array(self.bin_s),
            pop_names=np.array(self.pop_names),
            **{f"extra__{k}": v for k, v in self.extras.items()},
        )
        with open(path.with_suffix(".meta.json"), "w") as fh:
            json.dump(self.meta, fh, indent=2, sort_keys=True, default=str)
        return path.with_suffix(".npz")

    @classmethod
    def load(cls, path: Path | str) -> PopulationDataset:
        path = Path(path)
        npz_path = path if path.suffix == ".npz" else path.with_suffix(".npz")
        meta_path = npz_path.with_suffix("").with_suffix(".meta.json")
        with np.load(npz_path) as z:
            meta = {}
            if meta_path.exists():
                with open(meta_path) as fh:
                    meta = json.load(fh)
            return cls(
                x1=z["x1"],
                x2=z["x2"],
                behavior=z["behavior"],
                blocks=z["blocks"],
                bin_s=float(z["bin_s"]),
                pop_names=tuple(str(s) for s in z["pop_names"]),
                meta=meta,
                extras={k[len("extra__") :]: z[k] for k in z.files if k.startswith("extra__")},
            )
