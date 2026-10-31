# Changelog

## 0.2.1 — 2026-10-31
- Docs only. README cut to a landing page (install, quickstart, the three contracts, headline results); the full write-up with the verdict table and limits moved to `docs/results.md`. Fixed a dangling "question 2" reference, an unnamed paper, and two line-wrap artifacts the PyPI page exposed.

## 0.2.0 — 2026-10-31
- Experiment 3: D1 vs D2 striatal neurons, shared / private covariance and the lever, on the public Klee et al. (2026) two-photon dataset (six mice). Pre-registered (`docs/experiment3_preregistration.md`, `b1676da`) before the run; verdicts and effect sizes in the README. Loader (`klee_data.py`), count-matched draws, three-way partition, motion-regressed decoding, lag profile, `acd force`, `scripts/evaluate_prereg.py`.
- `PopulationDataset.extras` for per-bin covariates; `lagged.py` block-safe lag pairing and within-block shift null.
- README reframed around three output contracts (observations / state / action).
- Packaging: public API re-exported from `allen_circuit_dynamics`, `py.typed`, PyPI metadata and classifiers, `CITATION.cff`, `CONTRIBUTING.md`, `docs/DEVLOG.md` (was `RESTART.md`), `scripts/bootstrap_linux.sh` (was `blade_bootstrap.sh`), release workflow with PyPI trusted publishing.

## 0.1.1 — 2026-10-30
- First real-session run (Visual Coding Neuropixels 715093703); results in README.
- `acd fetch` chooses areas from the session (top VIS* by QC-passing units, `--min-units`, `--max-areas`) instead of a hard-coded list.
- Coupling penalty made scale-free: λ is dimensionless (scaled by trace(X'X)/K); grid 0–10. Permutation p-value and within-area share added to Experiment 2's metrics.
- Bootstrap: allensdk isolated in `.venv-allen`; pynwb/hdmf/boto3/six/dateutil pinned; parallel ranged S3 download (`scripts/pdl.sh`); post-fetch steps split into `scripts/run_experiments.sh`.

## 0.1.0 — 2026-10-28
- Shared/private block-sparse factor analysis (EM), RRR/CCA/FA baselines, grouped-CV decoding with shift null.
- Connectivity-penalised linear dynamics with shuffled-atlas control.
- Synthetic generators for both experiments; `acd demo`; 14 tests.
- AllenSDK loader for Visual Coding Neuropixels + Mouse Connectivity Atlas projection matrix (untested on real data).
