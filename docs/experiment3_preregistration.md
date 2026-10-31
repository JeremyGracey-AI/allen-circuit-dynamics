# Experiment 3 — pre-registration

Committed before any model was fit to this dataset. The commit that adds this
file is the timestamp; `outputs/force_*/metrics.json` records the SHA of the
code that produced each result. Anything below that turns out wrong is left in
place and answered in the results section, not edited.

## Data

Klee, Fernando-Peiris, Athalye, Costa, Peterka & Sippy (2026), *Science
Advances*, "Shared striatal neurons exhibit context-specific dynamics for
internally and externally driven actions." Zenodo 10.5281/zenodo.16333867
(CC BY 4.0), member `Data Structures/Switching_2p_Struc_Main.mat`.

Six mice, one session each, two-photon calcium imaging of anterior dorsal
striatum at 30 fps; every ROI labelled D1-SPN (1), D2-SPN (2) or
unclassified (0; dropped). A lever on a rotary encoder, displacement in mm,
sampled per frame; blocks of cued and self-paced presses. Also per frame:
video motion energy (`VidChangeZ`). No EMG; no cross-day tracking; those parts
of the posting's question are out of reach here and are said so.

Preprocessing follows the paper where it is stated: neuropil-corrected traces
(0.7 scalar, already applied in the file) → ΔF/F with a 60 s running
50th-percentile baseline → z-score across the session. Then the repo's usual
steps: contiguous CV blocks (12 per session), fold-safe scaling. Where the
lever trace is shorter than the imaging (three mice, 148–187 frames), both are
truncated to the common length from the start and the dropped count is logged.

## Model

Experiment 1's block-sparse factor analysis on `[D1, D2]`:

```
x_D1 = W1 z + V1 u1 + e1      z: shared (d_shared)
x_D2 = W2 z + V2 u2 + e2      u1, u2: private to one population (d_private)
                              e: independent per neuron
```

Three-way vocabulary, fixed here so it cannot drift: **shared** = variance
explained by `z` (correlated across both populations); **private** = variance
explained by `u_k` (correlated within one population, not across); **independent**
= the per-neuron residual `e`. Athalye et al. (2017) call our private +
independent together "private"; both numbers are reported so either reading is
available.

Dimensions by held-out log-likelihood under grouped CV, grid
`d_shared ∈ {1,2,3,4,6,8}`, `d_private ∈ {0,1,2,4}`. Bin = 3 frames (100 ms).

## Controls, fixed in advance

1. **Count matching.** D1 and D2 are subsampled to equal size (the smaller
   count) in 10 random draws; every number is the mean ± SD across draws.
   Factor-analysis dimensionality depends on N; the paper equalizes counts too.
2. **Circular-shift null** for decoding (within blocks), as in Experiment 1.
3. **Motion covariate.** Each decoding result is reported twice: raw, and after
   regressing video motion energy out of both populations inside the training
   fold. A result that survives only the raw version is attributed to movement
   of the animal, not to the lever.
4. **Context split.** Everything is run on all frames, then separately on cued
   and self-paced blocks (BlockType 1 / 2).
5. **No peeking.** The grid, bins, lags, and thresholds below were set from
   the paper's methods and from Experiment 1, not from this data.

## Hypotheses and predictions

| | Hypothesis | Prediction (per mouse unless stated) | Source of the prediction |
|---|---|---|---|
| H1 | D1 and D2 encode the lever equally | Held-out R² of lever displacement from count-matched D1-only vs D2-only differs by < 0.10 in ≥ 5 of 6 mice | Klee 2026: separate D1 / D2 decoders reach similar accuracy |
| H2 | The action lives in the shared subspace | Shared-subspace R² > shift-null p95 in ≥ 5 of 6 mice, and shared R² > (private-D1 + private-D2) R² in ≥ 4 of 6 | Athalye 2017: task-relevant shared variance carries control |
| H3 | Neural activity leads the lever | Lagged decoding of the lever from the shared latents peaks at a **negative** lag (neural before lever) of 100–700 ms in ≥ 4 of 6 mice; private subspaces peak at a smaller lead or show no peak above null | Klee 2026: population trajectories diverge before movement |
| H4 | Context changes the dynamics, not the split | Shared-variance fraction differs by < 0.05 between cued and self-paced in most mice, while the H3 lead differs by > 100 ms in most mice (exploratory, no direction) | Klee 2026: same neurons, context-specific dynamics |

Decision rule: a hypothesis is "supported" only if its count threshold is met;
otherwise it is reported as not supported, with the per-mouse numbers. There is
no third category.

## Metrics written to `metrics.json`

Per mouse, per subsample draw, per context (all / cued / self-paced):
selected `(d_shared, d_private)`; variance partition (shared / private /
independent) for D1 and D2; held-out R² of lever from shared, private-D1,
private-D2, all latents, raw D1, raw D2, raw both, with shift-null p95; lagged
decoding curve for shared and each private subspace over lags −45…+45 bins
(±1.5 s at 100 ms) with the lag of the peak; RRR cross-prediction D2 from D1
and D1 from D2 by rank; the same decoding set after motion regression.

## What this cannot show

One session per mouse, so nothing about learning across days. No EMG. Calcium
at 30 fps, so no claim about delays shorter than ~100 ms between D1 and D2;
the two populations are intermingled in one field of view and a lag between
them is not the question. "Lever displacement" is the measured output; the
posting's "force" is the isometric version of the same quantity in a
different task.
