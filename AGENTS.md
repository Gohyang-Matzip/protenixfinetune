# Repository Guidelines

## Project Structure & Module Organization

- `fragment_ft/` contains the Python CLI and implementation. `data.py` handles manifests, splits, and metrics; `sources.py` imports public observations; `training.py` implements task heads and checkpoints.
- `protenix_adapter.py` connects existing Protenix environments; `synthetic.py` implements the optional artificial-negative comparison.
- `tests/test_workflow.py` and `tests/test_general.py` cover legacy and general screening workflows.
- `examples/` contains fictional CSV/JSON inputs. `docs/DATA.md`, `docs/TRAINING.md`, and `docs/VALIDATION.md` describe policies, execution, and verified limits. Preserve historical documents in `docs/archive/`.

## Build, Test, and Development Commands

Run from the repository root with Python 3.11+. No build step is required.

```bash
python3 -m fragment_ft --help
python3 -m fragment_ft validate examples/manifest.csv
python3 -m unittest discover -s tests -v
```

These commands display CLI usage, validate an example manifest, and run the test suite. Use an existing PyTorch-enabled interpreter for training tests; otherwise they skip. Consult `docs/TRAINING.md` before running native preparation or training. Use new output paths because generated artifacts refuse overwrite.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` functions/modules, `PascalCase` classes, and uppercase constants. Match surrounding quote style and add concise docstrings for non-obvious behavior. Prefer standard-library solutions and keep ML dependencies optional for data-only commands. No formatter or linter is configured. Preserve legacy manifest and checkpoint compatibility where supported.

## Testing Guidelines

Use standard-library `unittest`, files named `test_*.py`, and methods named `test_<behavior>`. Add focused regressions for changed parsing, eligibility, split leakage, loss routing, or checkpoint behavior. Use small fixtures and temporary directories. No numeric coverage threshold is configured. Report skipped tests explicitly; CPU mock-backend success does not establish Protenix, CUDA, or distributed compatibility.

## Commit & Pull Request Guidelines

This workspace has no Git metadata, so historical commit conventions cannot be verified. Use concise imperative subjects, such as `Fix assay outcome mapping`. Keep PRs focused; describe the problem, behavior change, validation commands, and remaining limitations. Link relevant issues and update affected documentation/examples. Do not claim native execution without evidence.

## Data Safety & Agent Instructions

Never convert failed, unknown, or phenotypic observations into binding negatives. Preserve raw provenance and reviewed outcome mappings. Keep raw data, prepared tensors, runs, and checkpoints gitignored. Archive artifacts instead of destructively deleting them. Do not install Protenix or download weights implicitly. Stop immediately when instructed to stop.
