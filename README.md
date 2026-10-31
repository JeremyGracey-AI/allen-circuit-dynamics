# allen-circuit-dynamics

[![CI](https://github.com/JeremyGracey-AI/allen-circuit-dynamics/actions/workflows/ci.yml/badge.svg)](https://github.com/JeremyGracey-AI/allen-circuit-dynamics/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/allen-circuit-dynamics.svg?cacheSeconds=3600)](https://pypi.org/project/allen-circuit-dynamics/)
[![Python](https://img.shields.io/pypi/pyversions/allen-circuit-dynamics.svg?cacheSeconds=3600)](https://pypi.org/project/allen-circuit-dynamics/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Latent-variable models of simultaneously recorded neural populations, run on
open data from the Allen Institute and its collaborators, with a pipeline that
runs end-to-end on synthetic data in under a minute. Independent work; not
affiliated with or endorsed by the Allen Institute.

```bash
pip install allen-circuit-dynamics
acd demo --out outputs/demo          # both Neuropixels experiments on synthetic data, ~30 s, no downloads
```

```python
from allen_circuit_dynamics import PopulationDataset, SharedPrivateFA, run_latent_experiment

ds = PopulationDataset.load("my_session.npz")          # x1, x2, behavior, blocks, bin_s
metrics = run_latent_experiment(ds)                    # held-out selection, partition, decoding
model = SharedPrivateFA(d_shared=3, d_private=(2, 2)).fit(ds.x1, ds.x2)
latents = model.transform(ds.x1, ds.x2)                # {"shared": ..., "private1": ..., "private2": ...}
```

One model sits underneath: a population's activity is a few shared latents,
a few private latents, and per-neuron noise, possibly with linear dynamics on
the latents. The three experiments score that model on three different
**output contracts**, in the sense of asking what the model is *for*:

| | Contract | Scored by | Data |
|---|---|---|---|
| Exp 1 | **Observations** — reconstruct the recorded activity | held-out log-likelihood; shared / private / independent variance partition | Visual Coding Neuropixels, VISrl ↔ VISp |
| Exp 2 | **State** — predict the next latent state | held-out one-step R², with and without an anatomical prior | Visual Coding Neuropixels, five visual areas + Mouse Connectivity Atlas |
| Exp 3 | **Action** — predict the animal's output from the latents | held-out R² of lever displacement per subspace, lag profile, pre-registered | Klee et al. 2026 two-photon D1 / D2 striatum (Zenodo) |

A model that scores well on one contract need not score well on another; the
point of running all three is to say which one each dataset supports. Full
results, figures and limits: [`docs/results.md`](https://github.com/JeremyGracey-AI/allen-circuit-dynamics/blob/main/docs/results.md). (The
observation / state / action split is the agent–environment loop with the
latent as the state; the same three-way reading has been used to sort
"world models" by what they emit.)

## What came out

**Experiment 1 — observations.** Visual Coding session 715093703, VISrl ↔ VISp,
50 ms bins. Held-out likelihood selects 6 shared + 4 private dimensions. The
shared subspace decodes running speed (R² 0.27, circular-shift null p95
−0.11); neither private subspace does; cross-area prediction is near zero
(RRR ≤ 0.04). At this bin size the shared signal reads as slow common drive
(locomotion / arousal), not communication.

![real decoding](https://raw.githubusercontent.com/JeremyGracey-AI/allen-circuit-dynamics/main/outputs/latent_715093703/decoding_by_subspace.png)

**Experiment 2 — state.** Five visual areas, same session. The atlas prior does
no better than a shuffled prior at any penalty strength, and unconstrained
coupling norms do not track projection density (r = 0.38, permutation
p = 0.075). A null, reported as one. The first real run exposed two bugs —
a penalty that was not scale-free, and a correlation computed from the
penalised fit (which had read r = 0.54, p = 0.016) — both fixed and kept in
the record.

**Experiment 3 — action.** Klee et al. (2026), six mice, D1 and D2 striatal
neurons imaged together with a lever; pre-registered before the run
([`docs/experiment3_preregistration.md`](https://github.com/JeremyGracey-AI/allen-circuit-dynamics/blob/main/docs/experiment3_preregistration.md)).
In every mouse the two populations share one subspace with nearly identical
shared fractions (0.18 / 0.18 … 0.26 / 0.21) and 1–5 % population-specific
variance. The shared subspace decodes the lever in two mice (R² 0.36, 0.17)
and not in three; the predicted neural lead does not appear at 30 fps. Three
of six pre-registered predictions were met; all six are scored against the
written thresholds in `docs/results.md`.

![JK238 lag profile](https://raw.githubusercontent.com/JeremyGracey-AI/allen-circuit-dynamics/main/outputs/force_klee/JK238/force_lag_profile.png)

Every result is a first pass on one session per animal, not a study. The
limits are listed in [`docs/results.md`](https://github.com/JeremyGracey-AI/allen-circuit-dynamics/blob/main/docs/results.md#limits-stated-up-front).

## Models

```
x1 = W1 z + V1 u1 + mu1 + e1        z  : shared latents        (both populations)
x2 = W2 z + V2 u2 + mu2 + e2        u1, u2 : private latents   (one population each)
                                    e  : independent per neuron
```

Block-sparse factor analysis fit by EM with the zero pattern enforced at every
M-step — the static case of probabilistic CCA with explicit private subspaces
(Bach & Jordan 2005) and of DLAG without delays (Gokcen et al. 2022).
Dimensions by held-out log-likelihood under grouped cross-validation where
every fold is a run of contiguous time. Benchmarks: reduced-rank regression
(Semedo et al. 2019), CCA, per-population FA. Decoding by ridge against a
circular-shift null that keeps the behavior's autocorrelation.

For the state contract, per-area latents are stacked and fit with
`S[t+1] = S[t] A + b`; the atlas enters as a penalty on each area-to-area block
of `A` proportional to how weak that projection is, scaled to the data
curvature so λ is dimensionless; the control is the same fit with a shuffled
atlas.

## Run it

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
acd demo --out outputs/demo_local  # both experiments, synthetic, ~30 s (outputs/demo/ is the committed run)
pytest
```

Real data (Visual Coding Neuropixels, public S3 bucket, no credentials):

```bash
uv venv --python 3.10 .venv-allen && uv pip install --python .venv-allen/bin/python -e ".[allen]"   # allensdk pins old scipy/pandas
.venv-allen/bin/acd fetch --session 715093703 --out data/ecephys    # areas chosen from the session (top VIS* by QC-passing units)
SESSION=715093703 bash scripts/run_experiments.sh                    # connectivity matrix + both experiments
```

`scripts/bootstrap_linux.sh --fetch` does all of the above on a fresh Linux or WSL box,
including a parallel ranged download of the 2.7 GB NWB from Allen's public S3
bucket (`scripts/pdl.sh`; allensdk's own download times out on slow links).

Experiment 3 (Klee et al. 2026, 246 MB, CPU only, ~45 min per mouse):

```bash
# Data Structures/Switching_2p_Struc_Main.mat from https://zenodo.org/records/16333867
acd force --mat Switching_2p_Struc_Main.mat --mouse all --out outputs/force_klee
```

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
  synth.py            planted-structure generators (all three experiments)
  preprocess.py       unit filter, sqrt, fold-safe scaling
  models/shared_private.py   block-sparse FA by EM
  models/baselines.py        RRR, CCA, per-population FA
  models/coupling.py         connectivity-penalised linear dynamics + shuffle control
  decode.py           grouped-CV ridge decoding with a circular-shift null
  experiments.py      experiments 1 and 2 as pure functions → metrics dict
  report.py           figures from metrics, one function per experiment
  allen_data.py       AllenSDK loader (lazy import) + atlas projection matrix
  klee_data.py        Klee et al. 2026 two-photon loader (h5py; dF/F per the paper)
  lagged.py           lag pairing that never crosses CV blocks; within-block shift null
  experiments_force.py  Experiment 3 (count-matched draws, contexts, lag profile)
  cli.py              acd demo | fetch | latent | coupling | force
docs/results.md                       full write-up, verdict table, limits
docs/experiment3_preregistration.md   hypotheses, grid, controls, decision rules — committed before the run
tests/                19 tests (~10 s without the two marked slow), synthetic only
scripts/              bootstrap_linux.sh, run_experiments.sh, evaluate_prereg.py, replot_force.py
```

## References

- Bach & Jordan (2005). A probabilistic interpretation of canonical correlation analysis.
- Semedo, Zandvakili, Machens, Yu & Kohn (2019). Cortical areas interact through a communication subspace. *Neuron*.
- Gokcen, Jasper, Semedo, Zandvakili, Kohn, Machens & Yu (2022). Disentangling the flow of signals between populations of neurons. *Nature Computational Science*.
- Siegle et al. (2021). Survey of spiking in the mouse visual system reveals functional hierarchy. *Nature*. (Visual Coding Neuropixels dataset)
- Oh et al. (2014). A mesoscale connectome of the mouse brain. *Nature*. (Mouse Connectivity Atlas)
- Klee, Fernando-Peiris, Athalye, Costa, Peterka & Sippy (2026). Shared striatal neurons exhibit context-specific dynamics for internally and externally driven actions. *Science Advances*. Data: Zenodo 10.5281/zenodo.16333867, CC BY 4.0. (Experiment 3 dataset)
- Athalye, Ganguly, Costa & Carmena (2017). Emergence of coordinated neural dynamics underlies neuroprosthetic learning and skillful control. *Neuron*. (shared vs. private variance vocabulary)
