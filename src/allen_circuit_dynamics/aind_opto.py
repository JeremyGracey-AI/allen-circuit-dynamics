"""Loader for the Neuropixels Opto optotagging sessions (Lakunina et al. 2026).

Lakunina, Socha, Ladd, …, Siegle & Carandini (2026), *Nature Methods*,
"Neuropixels Opto: combining high-resolution electrophysiology and
optogenetics." Data: Allen Institute for Neural Dynamics open bucket
``s3://aind-open-data`` (CC BY 4.0, anonymous HTTP), the 2025 NWB-zarr
re-exports ``ecephys_<subject>_<date>_<time>_nwb_2025-*``; forty sessions,
listed in Code Ocean capsule 1593868.

Only the columns this experiment needs are read, over HTTP, straight from the
zarr store: the units table (spike times, ``default_qc``, ``brain_region``,
``predicted_cell_type`` — the paper's own optotagging verdict), the running
``Linear velocity`` series and the ``epochs`` table. Nothing else is
downloaded; a session costs ~100 MB of transfer, mostly spike times.

The result is an :class:`OptoSession`: spike counts per unit per bin over one
epoch (default ``pre-stimulus``, before any laser), the unit's cell-type label,
speed = |velocity| on the same bins, and contiguous CV blocks. ``population()``
turns two cell types into the :class:`PopulationDataset` the factor models
expect.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .dataset import PopulationDataset

BUCKET = "https://aind-open-data.s3.us-west-2.amazonaws.com"
STRIATUM = ("CP", "STR", "ACB", "STRd", "STRv")

# The forty sessions in the capsule, in its order.
SESSIONS: tuple[str, ...] = (
    "ecephys_655565_2023-04-03_13-34-16",
    "ecephys_655565_2023-04-05_16-17-25",
    "ecephys_655568_2023-05-01_15-26-47",
    "ecephys_655568_2023-05-03_15-21-12",
    "ecephys_655571_2023-05-09_13-53-48",
    "ecephys_655571_2023-05-15_13-39-49",
    "ecephys_655572_2023-05-09_15-03-29",
    "ecephys_655572_2023-05-15_15-06-36",
    "ecephys_660104_2023-07-24_18-18-13",
    "ecephys_661398_2023-03-31_17-01-09",
    "ecephys_661398_2023-04-03_15-47-29",
    "ecephys_661398_2023-04-06_16-02-15",
    "ecephys_662913_2023-07-20_16-15-45",
    "ecephys_666721_2023-05-09_11-01-03",
    "ecephys_666721_2023-05-12_16-15-36",
    "ecephys_666859_2023-06-13_14-19-34",
    "ecephys_666861_2023-05-23_10-04-20",
    "ecephys_666861_2023-05-24_14-30-05",
    "ecephys_671646_2023-07-31_15-21-40",
    "ecephys_674005_2023-08-08_15-39-38",
    "ecephys_674005_2023-08-10_13-15-14",
    "ecephys_678449_2023-08-28_17-53-41",
    "ecephys_678577_2023-11-06_15-58-49",
    "ecephys_682033_2023-10-26_14-21-51",
    "ecephys_682085_2023-10-03_16-42-49",
    "ecephys_684156_2023-10-11_17-45-08",
    "ecephys_684156_2023-10-13_16-56-59",
    "ecephys_692497_2023-12-11_16-20-55",
    "ecephys_692497_2023-12-15_16-29-04",
    "ecephys_694471_2023-12-11_14-51-57",
    "ecephys_694471_2023-12-15_15-26-01",
    "ecephys_694473_2023-11-28_15-23-23",
    "ecephys_697577_2024-02-13_16-24-10",
    "ecephys_704242_2024-04-30_17-13-29",
    "ecephys_704242_2024-05-02_16-51-50",
    "ecephys_715287_2024-04-24_15-24-43",
    "ecephys_715290_2024-05-09_15-39-23",
    "ecephys_715339_2024-05-08_16-09-35",
    "ecephys_719093_2024-05-13_16-42-41",
    "ecephys_719093_2024-05-15_15-01-10",
)


@dataclass(frozen=True)
class AindOptoConfig:
    session: str
    bin_s: float = 0.01
    n_blocks: int = 12
    epoch: str = "pre-stimulus"
    regions: tuple[str, ...] = STRIATUM
    qc_only: bool = True
    bucket: str = BUCKET


# ----------------------------------------------------------------------------- bucket access


def _list_prefixes(bucket: str, prefix: str) -> tuple[list[str], list[str]]:
    """One ListObjectsV2 page with ``delimiter=/``: (common prefixes, keys)."""
    url = f"{bucket}/?list-type=2&prefix={prefix}&delimiter=/&max-keys=1000"
    with urllib.request.urlopen(url, timeout=60) as resp:
        xml = resp.read().decode()
    return re.findall(r"<Prefix>([^<]*)</Prefix>", xml), re.findall(r"<Key>([^<]*)</Key>", xml)


def find_nwb(session: str, bucket: str = BUCKET) -> tuple[str, str]:
    """Return ``(asset_prefix, nwb_zarr_prefix)`` for the newest 2025 NWB re-export."""
    assets, _ = _list_prefixes(bucket, f"{session}_nwb")
    assets = sorted(a for a in assets if "_nwb_2025" in a)
    if not assets:
        raise FileNotFoundError(f"No 2025 NWB asset for {session} in {bucket}.")
    asset = assets[-1]
    inner, _ = _list_prefixes(bucket, asset)
    nwb = [p for p in inner if p.endswith(".nwb/")]
    if not nwb:
        raise FileNotFoundError(f"No .nwb zarr directory under {asset}.")
    return asset, nwb[0]


def _read_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.load(resp)


def open_nwb(bucket: str, nwb_prefix: str):
    """Open the NWB-zarr store read-only over HTTP (consolidated metadata)."""
    import fsspec
    import zarr

    store = fsspec.get_mapper(f"{bucket}/{nwb_prefix}")
    return zarr.open_consolidated(store, mode="r")


# ----------------------------------------------------------------------------- container


@dataclass(frozen=True)
class OptoSession:
    """Binned spikes for every included unit of one session, with cell-type labels.

    Attributes
    ----------
    counts:
        ``(n_bins, n_units)`` spike counts.
    unit_type:
        ``(n_units,)`` the file's ``predicted_cell_type`` (``D1``, ``D2``,
        ``ChAT``, …, ``untagged``).
    unit_id:
        ``(n_units,)`` row index into the NWB units table.
    speed:
        ``(n_bins,)`` |linear velocity| averaged into the bins.
    blocks:
        ``(n_bins,)`` contiguous CV block ids.
    """

    counts: np.ndarray
    unit_type: np.ndarray
    unit_id: np.ndarray
    speed: np.ndarray
    blocks: np.ndarray
    bin_s: float
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n, m = self.counts.shape
        if self.unit_type.shape != (m,) or self.unit_id.shape != (m,):
            raise ValueError("unit_type and unit_id must have one entry per unit.")
        if self.speed.shape != (n,) or self.blocks.shape != (n,):
            raise ValueError("speed and blocks must have one entry per bin.")

    @property
    def n_bins(self) -> int:
        return int(self.counts.shape[0])

    def types(self) -> dict[str, int]:
        t, c = np.unique(self.unit_type, return_counts=True)
        return {str(k): int(v) for k, v in zip(t, c, strict=True)}

    def units_of(self, cell_type: str) -> np.ndarray:
        return np.flatnonzero(self.unit_type == cell_type)

    def rebin(self, k: int) -> OptoSession:
        """Merge ``k`` consecutive bins (counts summed, speed averaged)."""
        n = (self.n_bins // k) * k
        c = self.counts[:n].reshape(n // k, k, -1).sum(axis=1)
        s = self.speed[:n].reshape(n // k, k).mean(axis=1)
        b = self.blocks[:n:k]
        return OptoSession(c, self.unit_type, self.unit_id, s, b, self.bin_s * k, dict(self.meta))

    def population(self, type1: str, type2: str) -> PopulationDataset:
        """Two cell types as a :class:`PopulationDataset` (counts as float)."""
        i1, i2 = self.units_of(type1), self.units_of(type2)
        return PopulationDataset(
            x1=self.counts[:, i1].astype(float),
            x2=self.counts[:, i2].astype(float),
            behavior=self.speed.astype(float),
            blocks=self.blocks,
            bin_s=self.bin_s,
            pop_names=(type1, type2),
            meta=dict(self.meta) | {"units_" + type1: i1.tolist(), "units_" + type2: i2.tolist()},
        )

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path.with_suffix(".npz"),
            counts=self.counts,
            unit_type=self.unit_type.astype(str),
            unit_id=self.unit_id,
            speed=self.speed,
            blocks=self.blocks,
            bin_s=np.array(self.bin_s),
        )
        with open(path.with_suffix(".meta.json"), "w") as fh:
            json.dump(self.meta, fh, indent=2)
        return path.with_suffix(".npz")

    @classmethod
    def load(cls, path: Path | str) -> OptoSession:
        path = Path(path)
        npz = path if path.suffix == ".npz" else path.with_suffix(".npz")
        meta_path = npz.with_suffix("").with_suffix(".meta.json")
        meta = {}
        if meta_path.exists():
            with open(meta_path) as fh:
                meta = json.load(fh)
        with np.load(npz) as z:
            return cls(
                counts=z["counts"],
                unit_type=z["unit_type"].astype(str),
                unit_id=z["unit_id"],
                speed=z["speed"],
                blocks=z["blocks"],
                bin_s=float(z["bin_s"]),
                meta=meta,
            )


# ----------------------------------------------------------------------------- loading


def _epoch_window(g, name: str) -> tuple[float, float]:
    ep = g["intervals/epochs"]
    tags = ep["tags"][:].astype(str)
    hit = np.flatnonzero(tags == name)
    if hit.size == 0:
        raise KeyError(f"epoch {name!r} not in {tags.tolist()}")
    i = int(hit[0])
    return float(ep["start_time"][i]), float(ep["stop_time"][i])


def load_aind_opto(cfg: AindOptoConfig, cache_dir: Path | str | None = None) -> OptoSession:
    """Fetch (or read from ``cache_dir``) one session as an :class:`OptoSession`."""
    cache = None
    if cache_dir is not None:
        tag = f"{cfg.session}_{cfg.epoch}_b{round(cfg.bin_s * 1000)}ms"
        cache = Path(cache_dir) / tag
        if cache.with_suffix(".npz").exists():
            return OptoSession.load(cache)

    asset, nwb = find_nwb(cfg.session, cfg.bucket)
    g = open_nwb(cfg.bucket, nwb)
    u = g["units"]

    region = u["brain_region"][:].astype(str)
    ctype = u["predicted_cell_type"][:].astype(str)
    qc = u["default_qc"][:].astype(bool)
    keep = np.isin(region, cfg.regions)
    if cfg.qc_only:
        keep &= qc
    unit_id = np.flatnonzero(keep)

    t0, t1 = _epoch_window(g, cfg.epoch)
    lv = g["processing/behavior/Linear velocity/linear velocity"]
    vt, vd = lv["timestamps"][:], lv["data"][:]
    t0 = max(t0, float(vt[0]))
    t1 = min(t1, float(vt[-1]))
    n_bins = int((t1 - t0) // cfg.bin_s)

    spike_times = u["spike_times"][:]
    index = u["spike_times_index"][:]
    starts = np.concatenate([[0], index[:-1]])
    counts = np.zeros((n_bins, unit_id.size), dtype=np.int32)
    for k, uid in enumerate(unit_id):
        st = spike_times[starts[uid] : index[uid]]
        st = st[(st >= t0) & (st < t1)]
        counts[:, k] = np.bincount(((st - t0) / cfg.bin_s).astype(int), minlength=n_bins)[:n_bins]

    m = (vt >= t0) & (vt < t1)
    which = ((vt[m] - t0) / cfg.bin_s).astype(int)
    num = np.bincount(which, weights=np.abs(vd[m]), minlength=n_bins)[:n_bins]
    den = np.bincount(which, minlength=n_bins)[:n_bins]
    speed = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    if np.isnan(speed).any():  # fill the odd empty bin from its neighbours
        ok = ~np.isnan(speed)
        speed = np.interp(np.arange(n_bins), np.flatnonzero(ok), speed[ok])

    blocks = np.minimum(np.arange(n_bins) * cfg.n_blocks // n_bins, cfg.n_blocks - 1)

    subject = _read_json(f"{cfg.bucket}/{asset}subject.json")
    details = subject.get("subject_details", subject)
    meta = {
        "source": "Lakunina et al. 2026 (Neuropixels Opto), AIND open data, CC BY 4.0",
        "bucket": cfg.bucket,
        "asset": asset,
        "nwb": nwb,
        "session": cfg.session,
        "subject_id": subject.get("subject_id"),
        "genotype": details.get("genotype"),
        "sex": details.get("sex"),
        "epoch": cfg.epoch,
        "epoch_window_s": [t0, t1],
        "bin_s": cfg.bin_s,
        "regions": list(cfg.regions),
        "qc_only": cfg.qc_only,
        "n_units_table": int(region.size),
        "n_units_striatal": int(np.isin(region, cfg.regions).sum()),
        "n_units_included": int(unit_id.size),
        "cell_types_included": {
            str(t): int(c)
            for t, c in zip(*np.unique(ctype[unit_id], return_counts=True), strict=True)
        },
        "behavior": "speed = |linear velocity| (running disc), bin mean",
        "speed_units": str(lv.attrs.get("unit", "")),
    }
    out = OptoSession(counts, ctype[unit_id], unit_id, speed, blocks, cfg.bin_s, meta)
    if cache is not None:
        out.save(cache)
    return out
