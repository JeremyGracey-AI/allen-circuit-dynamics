# Changelog

## 0.3.1 — 2026-11-01
- Docs and two post hoc scripts. No package code changed; every pre-registered verdict stands as
  scored. The corrections are dated and listed at the top of `docs/results.md`.
- Correction, Experiment 3. `scripts/posthoc_exp3_random_split.py` compares the D1/D2
  shared/private partition with 50 random two-way splits of the same neurons in each mouse. They
  match in all six mice (p 0.06–0.78), so the equal shared fractions describe the striatal
  population, not D1 versus D2. The 0.3.0 write-up called them "the robust finding"; that reading
  is withdrawn.
- Correction, Experiment 4. `scripts/posthoc_exp4_hierarchical.py` resamples mice, then sessions,
  then units. The lead's 95% intervals include zero (D1 −360 to +70 ms, 7 mice; D2 −260 to +60 ms,
  5 mice); per-mouse medians are negative in 5 of 7 and 4 of 5. The pre-registered unit-level H5a
  verdict stands; the write-up no longer reads it as a property of striatum. The statement that
  tagged units lead by more than untagged units is withdrawn (within-session difference −38 ms
  for D1, −20 ms for D2, no consistent sign).
- `docs/results.md`: both checks with tables, corrected readings, two new limits (units are not
  independent; the wheel-speed signal's processing delay is unverified). README matches.

## 0.3.0 — 2026-10-31
- Experiment 4: does striatal spiking lead the motor output? `acd lead` (new `aind` extra:
  `pip install "allen-circuit-dynamics[aind]"`) reads the Lakunina et al. (2026) Neuropixels Opto
  sessions straight from the Allen Institute for Neural Dynamics open bucket as NWB-zarr, and
  tests whether optotagged D1 / D2 units lead running speed: per-unit and population lag
  profiles against a within-block shift null, pooled H5–H7 verdicts. Pre-registered in
  `docs/experiment4_preregistration.md` before the first fit.
- Results: 31 of 40 sessions, 19 mice (5 sessions lack `brain_region`, 4 have no wheel).
  Speed-coupled D1 units lead running by a median 145 ms (bootstrap 95% CI 75–205 ms), D2 by
  100 ms (50–150 ms). H5a (lead) and H7 (coupling) supported; H5b (population ridge profile)
  and H6 (D1 = D2) not. Write-up in `docs/results.md`; `scripts/evaluate_prereg_exp4.py`
  rebuilds the verdict table from the per-session files.
- Post hoc population measure (mean z-rate cross-correlation), added after the first session
  showed the pre-registered ridge profile is dominated by block-level changes in running.
  Reported alongside; the verdicts use the pre-registered measure.
- Cross-machine check: every session computed twice (cloud, plus WSL2 or macOS) on different
  numpy / scipy / scikit-learn versions; all 55,298 recorded fields agree to 5e-13 and the
  pooled verdict file is identical.
- `acd lead` runs one session at a time (fetch, analyse, write), retries transient network
  errors (a missing column is not retried), records skipped sessions with reasons in the pooled
  `metrics.json`, and line-buffers its output so `tail -f` on a nohup log shows progress.
- `xcorr_lags` copies its sliding window to contiguous memory before the matmul; under numpy
  1.23 the strided-view product took a non-BLAS path (17 s vs 0.14 s per call). Results
  unchanged to 3e-14.
- Public API adds `OptoSession`, `LeadConfig`, `run_lead_session`, `pool_lead`.

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
