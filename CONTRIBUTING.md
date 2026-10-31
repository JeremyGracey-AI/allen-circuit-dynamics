# Contributing

## Setup

```bash
git clone https://github.com/JeremyGracey-AI/allen-circuit-dynamics.git
cd allen-circuit-dynamics
uv venv --python 3.12 && uv pip install -e ".[dev]"
.venv/bin/pytest -q -m "not slow"     # ~10 s
.venv/bin/pytest -q                   # ~1 min, includes the end-to-end synthetic runs
.venv/bin/ruff check src tests scripts && .venv/bin/ruff format --check src tests scripts
```

CI runs the lint, format check and full test suite on Python 3.10 and 3.12.

The AllenSDK loader (`acd fetch`) needs the `allen` extra in its **own** venv —
allensdk pins old scipy / pandas / numpy. `scripts/bootstrap_linux.sh` sets both
environments up; see the comments there for the pins and why they exist.

## Layout

- `src/allen_circuit_dynamics/models/` — the models, each a small class with
  `fit` / `score` / `transform`, no I/O.
- `src/allen_circuit_dynamics/experiments*.py` — an experiment is a pure
  function from a `PopulationDataset` (or arrays) and a frozen config
  dataclass to a JSON-serialisable metrics dict. No plotting, no file writes.
- `report.py` — one function per experiment that turns a metrics dict into figures.
- `cli.py` — argument parsing and the provenance block; nothing else.
- `synth.py` — a planted-structure generator per experiment; every experiment
  has a test that recovers what was planted.

## Adding an experiment

1. Generator in `synth.py` with the structure planted explicitly in the config.
2. `run_<name>_experiment(ds, cfg) -> dict` in a new `experiments_<name>.py`.
3. A recovery test, plus a `@pytest.mark.slow` end-to-end test if it takes more than a few seconds.
4. `plot_<name>` in `report.py`, a subcommand in `cli.py`, and a README section
   that states the contract the experiment scores (observations / state / action).
5. If it will run on real data, write the pre-registration first
   (`docs/experiment3_preregistration.md` is the template) and commit it before
   the run; the commit is the timestamp.

## Conventions

- Held-out numbers only. Cross-validation groups are contiguous time blocks.
- Every penalty is dimensionless (scaled to the data curvature). Experiment 2
  was wrong about this once; `docs/DEVLOG.md` has the story.
- A null that preserves autocorrelation (circular shift within blocks), never a
  plain permutation, for anything decoded from a smooth signal.
- Results go in `metrics.json` with a provenance block (package version, git
  SHA, library versions, config hash, dataset metadata). Figures are derived
  from the metrics, never the other way round.
- Numbers quoted in the README come from a committed `metrics.json`.

## Releasing

Bump `version` in `pyproject.toml` and `CITATION.cff`, add a CHANGELOG entry,
tag `vX.Y.Z` and push the tag. `.github/workflows/release.yml` builds and
publishes to PyPI through trusted publishing and creates a GitHub release.
