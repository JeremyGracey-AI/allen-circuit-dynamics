# Development log

Decisions and state, newest first. Kept so the next person (or the author in a month) knows why things are the way they are.

## 2026-11-01

- Two post hoc checks after 0.3.0, both scripts under `scripts/posthoc_*`, labelled post hoc in `docs/results.md`. Exp 3: the D1/D2 partition equals the partition of 50 random splits of the same neurons in all six mice (p 0.06–0.78), so "equal shared fraction" is a population property, not a cell-type finding. Exp 4: bootstrap over mice → sessions → units gives D1 [−360, +70] ms (7 mice), D2 [−260, +60] ms (5 mice); per-mouse medians negative in 5/7 and 4/5. Within-session tagged minus untagged: −38 ms D1 (5/10 earlier), −20 ms D2 (3/5); the 0.3.0 "tagged lead more" sentence is withdrawn.
- 0.3.1 is docs plus the two scripts. No package code changed; every pre-registered verdict stands as scored.
- Lesson for the next pre-registration: name the unit of inference (mouse, not neuron) and include a permutation control for any claim about cell type.

## 2026-10-31 (evening)

- Experiment 4 built and pre-registered (`0fe8bca`): spike-resolution lead test on the Neuropixels Opto optotagged D1/D2 units, read from `s3://aind-open-data` (2025 NWB-zarr re-exports carry `predicted_cell_type`). Loader `aind_opto.py`, runner `experiments_lead.py`, `acd lead`, 3 tests.
- First real session (`661398_2023-04-06`, 27 D1 / 4 D2): per-unit peak lags −0.24 s (D1), −0.37 s (D2), −0.16 s (untagged); mean-rate cross-correlation peaks −0.22 / −0.19 / −0.17 s, all above null. The pre-registered ridge lag profile is negative at every lag: the mouse runs in 3 of 12 CV blocks and sits in the rest, so held-out R² across contiguous blocks measures non-stationarity, not timing. Reported as "H5b not supported" with that explanation; the cross-correlation is labelled post hoc.
- Blade replication (first 22 sessions): identical to the cloud to 3e-13 across numpy 1.23 vs 2.4. The Blade's fetch phase skipped 3 sessions (655571_2023-05-09, 666861_2023-05-23, 674005_2023-08-08) that the cloud analysed; `acd lead` now retries transient errors and writes skips into the pooled metrics. Filled on macOS with `--resume` (9 sessions in ~2 min); all 31 committed sessions match the cloud run to 5e-13 and the pooled file is identical.
- Full run done in the cloud: 31 sessions analysed. D1 −145 ms [−205, −75], D2 −100 ms [−150, −50]; H5a + H7 supported, H5b + H6 not. Blade run (same code, seed) is the replication; diff its pooled `metrics.json` against the committed one.
- Full 40-session run is to happen on the Blade: `uv sync --extra aind && .venv/bin/acd lead --out outputs/lead_aind --resume` (~2 min per session).

## 2026-10-31

- Experiment 3 (D1/D2 striatum + lever, Klee et al. 2026, six mice) built, pre-registered, run in the cloud, written up; results under `outputs/force_klee/`, verdict table in README. Robust finding: equal shared fraction for D1 and D2 in every mouse, private 1–5% (reading withdrawn 2026-11-01: random splits give the same partition). Lever decoding substantive in 2 of 6 mice; H3 (neural lead) not supported at 30 fps.
- First real-session run (WSL2 Ubuntu 22.04): session 715093703, areas VISrl/VISp/VISpm/VISl/VISam, 12,009 bins. Results are in the README (one paragraph per experiment).
- 19 tests; `acd demo` reproduces the synthetic reference run in `outputs/demo/`.
- Experiment 2's λ is now dimensionless (penalty scaled to the data curvature) after the first real run showed the old absolute λ grid did nothing on a 12k-bin session.

## Next
1. Experiment 3 power analysis: plant D1- or D2-specific covariance in synthetic data matched to each mouse (neuron counts, bins, noise) and find the smallest private fraction the random-split control detects. If plausible effects are undetectable, the partition null is uninformative and leaves the write-up. Widen both grids first (`d_shared` and `d_private` ceilings hit in 4/6 mice).
2. Experiment 4: check the wheel-speed signal's processing delay against the raw encoder; look for more optotagged striatal sessions on `s3://aind-open-data`; dated pre-registration amendment that makes mouse-level statistics primary before any re-run.
3. Experiment 3, other follow-ups: deconvolve or use spike-inferred traces before any timing claim; event-aligned (press-triggered) decoding; DANDI 000404 (Athalye BMI) for the state contract.
4. Second session + 20 ms bins for Experiments 1–2 — the two obvious robustness checks before any sentence in the README is generalised.
5. DLAG-style time-delayed shared latents: the test that separates "slow common signal" from "communication" in Experiment 1.
6. Experiment 2 input term (stimulus / running speed as exogenous input) so the one-step model is not forced to explain stimulus-driven variance as recurrence.

## allensdk
- pynwb and hdmf are pinned (<2.6 / <3.14): newer ones cannot open the NWB 2.2.2 files (`Can't instantiate abstract class NWBFile`), and the allensdk cache then mistakes the reader error for a missing file and re-downloads. The bootstrap opens the NWB once before calling the cache for exactly this reason.
- The NWB itself comes from Allen's public S3 bucket via curl (resumable); allensdk's own fetch times out at 1200 s on slow links. If the manifest's expected path ever changes, print `cache.get_session_table()` and compare.
- Lives in its own `.venv-allen` (Python 3.10) because it pins scipy<1.11 / pandas<2.1 / numpy<2; the bootstrap script does this. Only `acd fetch` and the connectivity matrix run there; the `.npz` outputs are plain numpy and `.venv` runs the experiments.

## Why these choices (for the write-up)
- Block-sparse FA, not DLAG: DLAG adds time delays and is the published alternative to benchmark against *next*; this is the static baseline it should beat.
- RRR as the yardstick for shared dimensionality is the standard from Semedo 2019.
- Circular-shift null, not permutation, because running speed is autocorrelated.
