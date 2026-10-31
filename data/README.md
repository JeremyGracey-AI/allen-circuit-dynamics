# data/

Nothing in here is committed except this file.

- `ecephys_cache/` — AllenSDK `EcephysProjectCache` manifest + NWB files (1–3 GB per session). Created by `acd fetch`.
- `ecephys/<session>_<A>_<B>.npz` + `.meta.json` — the two-population dataset for Experiment 1.
- `ecephys/<session>_areas.npz` — per-area PCA latents for Experiment 2.
- `conn_VIS6.npy` — row-normalised projection-density matrix from the Mouse Connectivity Atlas.

`scripts/bootstrap_linux.sh --fetch` produces all of them.
