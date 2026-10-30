# allen-circuit-dynamics

Two questions about simultaneously recorded neural populations, asked of
Allen Institute open Neuropixels data, with a pipeline that runs end-to-end on
synthetic data in under a minute.

1. **Shared vs. private.** When two populations are recorded at the same time,
   how much of each one's activity is shared with the other, how much is
   private, and which part encodes a continuously measured behavior?
2. **Does wiring constrain dynamics?** When several areas are recorded together,
   does the anatomical projection structure between them predict the
   inter-area coupling of a linear dynamical model fit to the activity?

Status: pipeline complete and tested on synthetic data; real-data runs pending
(the NWB files live on a separate machine). No results on real recordings are
claimed in this README yet.

## Experiment 1 — shared / private factor analysis

Model: factor analysis on the concatenated vector `[x1, x2]` with a
block-sparse loading matrix, fit by EM with the zero pattern enforced at every
M-step.

```
x1 = W1 z + V1 u1 + mu1 + e1        z  : shared latents (d_shared)
x2 = W2 z + V2 u2 + mu2 + e2        u1 : private to population 1 (d_private)
                                    u2 : private to population 2 (d_private)
```

It is the static special case of probabilistic CCA with explicit private
subspaces (Bach & Jordan 2005) and of DLAG without time delays
(Gokcen et al. 2022). The experiment:

- selects `d_shared` and `d_private` by **held-out log-likelihood** under
  block-grouped cross-validation (contiguous time stays together),
- reports the **variance partition** (shared / private / noise) per population,
- benchmarks cross-population prediction (E[x2 | x1]) against
  **reduced-rank regression** at each rank (Semedo et al. 2019) and CCA,
- **decodes the behavior** from each subspace (shared, private-1, private-2,
  raw populations) with ridge regression under grouped CV, against a
  circular-shift null that preserves the behavior's autocorrelation.

On the synthetic demo (planted: 3 shared, 2 private per population, behavior a
readout of the shared factors plus population-1's private factors) the shared
subspace carries the behavior (R² ≈ 0.74), population-2's private subspace does
not (≈ 0), and the RRR curve flattens by rank 3. At the demo's firing rates the
linear-Gaussian model selects more dimensions than were planted (6 shared,
4 private) because Poisson counts put nonlinear structure into extra linear
directions; in the lower-gain regime the test suite uses, held-out likelihood
selects the planted 3 exactly. See Limits.

![decoding](outputs/demo/latent/decoding_by_subspace.png)

## Experiment 2 — connectivity-constrained linear dynamics

Per-area population latents are stacked into `S` and fit with a one-step
linear model `S[t+1] = S[t] A + b`. The atlas enters as a penalty on each
area-to-area block of `A` proportional to how *weak* that projection is:

```
penalty(A) = λ · Σ_{a,b} (1 − conn[a,b]) · ‖A_block(a,b)‖²
```

The test is held-out one-step prediction of the atlas-constrained model vs. an
unconstrained ridge model vs. models fit with a **shuffled** connectivity
matrix at the same λ. If the atlas carries information about the dynamics the
true-atlas curve should separate from the shuffled curve as λ grows; if it
doesn't, that's the finding. On the synthetic demo (six areas, eight latents
each, 1 200 bins, a planted sparse projection matrix) the two curves separate
at large λ and the fitted block norms correlate with the planted connectivity.

![coupling](outputs/demo/coupling/coupling_vs_lambda.png)

## Run it

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
acd demo --out outputs/demo        # both experiments, synthetic, ~30 s
pytest
```

Real data (Visual Coding Neuropixels, public S3 bucket, no credentials):

```bash
uv venv --python 3.10 .venv-allen && uv pip install --python .venv-allen/bin/python -e ".[allen]"   # allensdk pins old scipy/pandas
.venv-allen/bin/acd fetch --session 715093703 --areas VISp VISl VISal VISpm VISam VISrl --out data/ecephys
acd latent   --data data/ecephys/715093703_VISp_VISl.npz --out outputs/latent_715093703
acd coupling --data data/ecephys/715093703_areas.npz --conn data/conn_VIS6.npy --out outputs/coupling_715093703
```

`scripts/blade_bootstrap.sh --fetch` does all of the above on a fresh WSL box,
including building the connectivity matrix from the Mouse Connectivity Atlas.

## What the loader does to the data

- Unit QC: `isi_violations < 0.5`, `amplitude_cutoff < 0.1`, `presence_ratio > 0.9`.
- Bins spikes at 50 ms inside the chosen stimulus epoch only (default
  `natural_movie_one`); a bin never straddles two presentation windows.
- Behavior = mean running speed per bin; bins with no speed samples are
  dropped and counted in the metadata.
- Units under 0.5 Hz are dropped and counted. Counts are square-root
  transformed, then z-scored with training-fold statistics inside every
  cross-validation loop.
- Twelve contiguous blocks define the CV groups.

Every `metrics.json` carries a provenance block: package version, git SHA,
library versions, config hash, and the dataset's own metadata.

## Layout

```
src/allen_circuit_dynamics/
  dataset.py          PopulationDataset — the one container every stage uses
  synth.py            planted-structure generators (both experiments)
  preprocess.py       unit filter, sqrt, fold-safe scaling
  models/shared_private.py   block-sparse FA by EM
  models/baselines.py        RRR, CCA, per-population FA
  models/coupling.py         connectivity-penalised linear dynamics + shuffle control
  decode.py           grouped-CV ridge decoding with a circular-shift null
  experiments.py      the two experiments as pure functions → metrics dict
  report.py           one figure per question
  allen_data.py       AllenSDK loader (lazy import) + atlas projection matrix
  cli.py              acd demo | fetch | latent | coupling
tests/                14 tests, ~30 s, synthetic only
scripts/blade_bootstrap.sh
```

## Limits, stated up front

- The shared/private model is linear-Gaussian and static. On Poisson-like
  counts it needs more dimensions than the generative truth to absorb the
  nonlinearity; held-out likelihood, not the planted number, is the criterion.
- Experiment 2's "areas" are PCA latents, not cell types, and the atlas prior
  is projection *density*, which says nothing about sign or cell-type
  specificity. The experiment can only say whether that coarse structure
  predicts coupling — not why.
- Running speed is a behavioral proxy available in the open dataset; it is not
  a measured motor output. The code accepts any continuous signal on the same
  bins.
- Nothing here has been run on real recordings yet. See Status above.

## References

- Bach & Jordan (2005). A probabilistic interpretation of canonical correlation analysis.
- Semedo, Zandvakili, Machens, Yu & Kohn (2019). Cortical areas interact through a communication subspace. *Neuron*.
- Gokcen, Jasper, Semedo, Zandvakili, Kohn, Machens & Yu (2022). Disentangling the flow of signals between populations of neurons. *Nature Computational Science*.
- Siegle et al. (2021). Survey of spiking in the mouse visual system reveals functional hierarchy. *Nature*. (Visual Coding Neuropixels dataset)
- Oh et al. (2014). A mesoscale connectome of the mouse brain. *Nature*. (Mouse Connectivity Atlas)
