# Development log

Decisions and state, newest first. Kept so the next person (or the author in a month) knows why things are the way they are.

## 2026-10-31 (evening)

- Experiment 4 built and pre-registered (`0fe8bca`): spike-resolution lead test on the Neuropixels Opto optotagged D1/D2 units, read from `s3://aind-open-data` (2025 NWB-zarr re-exports carry `predicted_cell_type`). Loader `aind_opto.py`, runner `experiments_lead.py`, `acd lead`, 3 tests.
- First real session (`661398_2023-04-06`, 27 D1 / 4 D2): per-unit peak lags −0.24 s (D1), −0.37 s (D2), −0.16 s (untagged); mean-rate cross-correlation peaks −0.22 / −0.19 / −0.17 s, all above null. The pre-registered ridge lag profile is negative at every lag: the mouse runs in 3 of 12 CV blocks and sits in the rest, so held-out R² across contiguous blocks measures non-stationarity, not timing. Reported as "H5b not supported" with that explanation; the cross-correlation is labelled post hoc.
- Blade replication (first 22 sessions): identical to the cloud to 3e-13 across numpy 1.23 vs 2.4. The Blade's fetch phase skipped 3 sessions (655571_2023-05-09, 666861_2023-05-23, 674005_2023-08-08) that the cloud analysed; `acd lead` now retries transient errors and writes skips into the pooled metrics. Rerun with `--resume` to fill them.
- Full run done in the cloud: 31 sessions analysed. D1 −145 ms [−205, −75], D2 −100 ms [−150, −50]; H5a + H7 supported, H5b + H6 not. Blade run (same code, seed) is the replication; diff its pooled `metrics.json` against the committed one.
- Full 40-session run is to happen on the Blade: `uv sync --extra aind && .venv/bin/acd lead --out outputs/lead_aind --resume` (~2 min per session).

## 2026-10-31

- Experiment 3 (D1/D2 striatum + lever, Klee et al. 2026, six mice) built, pre-registered, run in the cloud, written up; results under `outputs/force_klee/`, verdict table in README. Robust finding: equal shared fraction for D1 and D2 in every mouse, private 1–5%. Lever decoding substantive in 2 of 6 mice; H3 (neural lead) not supported at 30 fps.
- First real-session run (WSL2 Ubuntu 22.04): session 715093703, areas VISrl/VISp/VISpm/VISl/VISam, 12,009 bins. Results are in the README (one paragraph per experiment).
- 19 tests; `acd demo` reproduces the synthetic reference run in `outputs/demo/`.
- Experiment 2's λ is now dimensionless (penalty scaled to the data curvature) after the first real run showed the old absolute λ grid did nothing on a 12k-bin session.

## Next
1. Experiment 3 follow-ups: widen the `d_shared` grid (ceiling hit in 4/6 mice); deconvolve or use spike-inferred traces before any timing claim; event-aligned (press-triggered) decoding to complement the continuous one; DANDI 000404 (Athalye BMI) for the state contract.
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
