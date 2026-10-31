# Experiment 4 — pre-registration

Committed before any model was fit to this dataset. The commit that adds this
file is the timestamp; `outputs/lead_aind/**/metrics.json` records the SHA of
the code that produced each result. Anything below that turns out wrong is left
in place and answered in the results section, not edited.

## Why

Experiment 3 predicted that striatal activity leads the motor output (H3) and
found no lead at 30 fps calcium imaging. The write-up blamed the calcium
kernel. That is an excuse until it is tested on spikes. This experiment tests
it on spikes from the same two cell types.

## Data

Lakunina, Socha, Ladd, …, Siegle & Carandini (2026), *Nature Methods*,
"Neuropixels Opto: combining high-resolution electrophysiology and
optogenetics." Public data: Allen Institute for Neural Dynamics bucket
`s3://aind-open-data` (CC BY 4.0, no account), the 2025 NWB-zarr re-exports
`ecephys_<subject>_<date>_<time>_nwb_2025-*`, forty sessions in twenty-six
mice, listed in the Code Ocean capsule 1593868.

Per session the NWB carries a units table with spike times, Kilosort 2.5 +
SpikeInterface quality metrics (`default_qc`), CCF `brain_region`, and the
paper's own optotagging verdict per unit (`predicted_cell_type`: `D1`, `D2`,
`ChAT`, …, `untagged`) with the per-colour latency / reliability / significant
pulse counts behind it. The tagging criteria are the paper's (≥ 4 of 5 pulses
significant, latency < 8 ms, reliability > 0.3; red-tagged wins if both); they
are read from the file, not re-derived. Behaviour is `processing/behavior/
Linear velocity` (running disc, 100 Hz). Epochs split each session into
`pre-stimulus` (611 s or 1211 s, no laser) and `stimulus`.

Survey before any fit (QC-passing units in CP/STR, from the file's own
columns): tagged D1 units exist in 16 sessions (1–27 per session, ~136 total),
tagged D2 in 12 sessions (1–19, ~70 total). Both types tagged with ≥ 3 units
each in **one** session only, `ecephys_661398_2023-04-06` (27 D1, 4 D2). That
sets what can be asked.

Inclusion: QC-passing (`default_qc`), `brain_region` in {CP, STR, ACB, STRd,
STRv}, pre-stimulus epoch only. A session enters the per-type analysis for a
cell type if it has ≥ 3 such units of that type.

Preprocessing: spike counts in 10 ms bins; counts and speed smoothed with the
same symmetric Gaussian (σ = 25 ms) so smoothing cannot shift a peak; speed =
|linear velocity| resampled to the bins by mean; 12 contiguous CV blocks over
the epoch; fold-safe scaling. Behaviour here is running on a disc, not a lever;
that is a different motor output than Experiment 3 and is said so wherever a
comparison is drawn.

## Measurements

1. **Per-unit lag profile.** Pearson correlation between a unit's smoothed
   rate and speed at lags −100…+100 bins (±1 s), `lag = t_neural − t_speed`
   (negative: neural leads). Null: 200 circular shifts of speed within blocks,
   max |r| over lags per shift; a unit is **speed-coupled** if its peak |r|
   exceeds the null's 95th percentile. Peak lag is the lag of max |r|.
2. **Population lag profile.** Experiment 3's `lag_profile`: held-out ridge
   R² of speed from the tagged population of one type at each lag, circular-
   shift null, peak lag and whether the peak clears the null. Also run for a
   random 50-unit subset of untagged striatal units per session, as the
   reference population.
3. **Exploratory, one session.** On `661398_2023-04-06` only, Experiment 3's
   shared/private factor analysis on [D1, D2] at 100 ms bins (to match
   Experiment 3), grid `d_shared ∈ {1, 2}`, `d_private ∈ {0, 1}`, and the lag
   profile of the shared latents. Four D2 units; reported, never generalised.

## Hypotheses and predictions

| | Hypothesis | Prediction | Source |
|---|---|---|---|
| H5a | Striatal spiking leads running | Pooled across sessions, the median peak lag of speed-coupled units is between −500 and −50 ms, with a 1000-draw bootstrap 95% CI of the median that excludes 0, for D1 **and** for D2 | Exp 3 H3's premise; Klee 2026 trajectories diverge before movement |
| H5b | …at the population level too | Population lag-profile peak is negative and above null in ≥ 2/3 of eligible sessions, per type | same |
| H6 | D1 and D2 lead by the same amount | Bootstrap 95% CI of (median lag D1 − median lag D2) lies within ±100 ms | Klee 2026: both types encode the action; Exp 3 H1 |
| H7 | Both types are speed-coupled | Fraction of speed-coupled units ≥ 0.30 for D1 and for D2 | Klee 2026 |

Decision rule as in Experiment 3: "supported" only if the threshold is met;
otherwise "not supported" with the numbers. If H5a fails, Experiment 3's
calcium-kernel explanation for its H3 is withdrawn in the write-up, not
softened.

## Metrics written to `metrics.json`

Per session: subject, genotype, epoch length, unit counts by type before and
after inclusion; per unit (type, peak lag, peak r, null p95, coupled); per type
and for the untagged reference: population lag profile (lags, R², null p95,
peak lag, above-null). Pooled file: per-type lists of coupled-unit peak lags,
medians with bootstrap CIs, the D1−D2 difference CI, coupling fractions, and
the H5–H7 verdicts.

## What this cannot show

Running is not a lever press; a lead here says the striatum precedes locomotor
output, not force. Tagged counts are small per session, so H5b is a weak test
and H5a (pooled) is the primary one. One session has both types; nothing about
a shared subspace generalises from it. Pre-stimulus epochs only, so 10–20 min
per session.
