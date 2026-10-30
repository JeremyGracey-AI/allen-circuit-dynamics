# RESTART — allen-circuit-dynamics

State as of 2026-10-28 (built in a cloud session, delivered to the Mac Studio):

- Pipeline complete; `acd demo` runs both experiments on synthetic data (~30 s); 14 tests pass.
- No real-data run yet. The Blade 18 (WSL2 Ubuntu-22.04, RTX 5090) is the box for it.

## Next on the Blade
1. `git clone` this repo into `D:\dev\allen-circuit-dynamics` (Windows) or `~/src/github.com/JeremyGracey-AI/allen-circuit-dynamics` (WSL ext4 — faster).
2. `bash scripts/blade_bootstrap.sh --fetch` — installs allensdk, pulls session 715093703 (~2 GB), builds the VIS6 connectivity matrix, runs both experiments.
3. Look at `outputs/latent_715093703/*.png` and `outputs/coupling_715093703/*.png`; copy them into `outputs/real/` and update the README Status + one results paragraph per experiment. Keep the "Limits" section.
4. Open the repo (it's private until the real-data run is in).

## allensdk
- Lives in its own `.venv-allen` (Python 3.10) because it pins scipy<1.11 / pandas<2.1 / numpy<2; the bootstrap script does this. Only `acd fetch` and the connectivity matrix run there; the `.npz` outputs are plain numpy and `.venv` runs the experiments.

## Why these choices (for the write-up)
- Block-sparse FA, not DLAG: DLAG adds time delays and is the published alternative to benchmark against *next*; this is the static baseline it should beat.
- RRR as the yardstick for shared dimensionality is the standard from Semedo 2019.
- Circular-shift null, not permutation, because running speed is autocorrelated.
