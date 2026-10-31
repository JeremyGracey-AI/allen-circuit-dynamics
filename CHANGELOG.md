# Changelog

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
