# RESTART — allen-circuit-dynamics

State as of 2026-10-30 (evening):

- First real-session run done on the Blade 18 (WSL2 Ubuntu-22.04): session 715093703, areas VISrl/VISp/VISpm/VISl/VISam, 12,009 bins. Results are in the README (one paragraph per experiment).
- Repo is public. 14 tests pass; `acd demo` reproduces the synthetic reference run in `outputs/demo/`.
- Experiment 2's λ is now dimensionless (penalty scaled to the data curvature) after the first real run showed the old absolute λ grid did nothing on a 12k-bin session.

## Next
1. Done: real-session outputs are committed under `outputs/*_715093703/` and quoted in the README.
2. Second session + 20 ms bins — the two obvious robustness checks before any sentence in the README is generalised.
3. DLAG-style time-delayed shared latents: the test that separates "slow common signal" from "communication" in Experiment 1.
4. Experiment 2 input term (stimulus / running speed as exogenous input) so the one-step model is not forced to explain stimulus-driven variance as recurrence.

## allensdk
- pynwb and hdmf are pinned (<2.6 / <3.14): newer ones cannot open the NWB 2.2.2 files (`Can't instantiate abstract class NWBFile`), and the allensdk cache then mistakes the reader error for a missing file and re-downloads. The bootstrap opens the NWB once before calling the cache for exactly this reason.
- The NWB itself comes from Allen's public S3 bucket via curl (resumable); allensdk's own fetch times out at 1200 s on slow links. If the manifest's expected path ever changes, print `cache.get_session_table()` and compare.
- Lives in its own `.venv-allen` (Python 3.10) because it pins scipy<1.11 / pandas<2.1 / numpy<2; the bootstrap script does this. Only `acd fetch` and the connectivity matrix run there; the `.npz` outputs are plain numpy and `.venv` runs the experiments.

## Why these choices (for the write-up)
- Block-sparse FA, not DLAG: DLAG adds time delays and is the published alternative to benchmark against *next*; this is the static baseline it should beat.
- RRR as the yardstick for shared dimensionality is the standard from Semedo 2019.
- Circular-shift null, not permutation, because running speed is autocorrelated.
