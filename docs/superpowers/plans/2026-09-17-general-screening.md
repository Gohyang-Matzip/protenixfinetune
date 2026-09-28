# General screening implementation plan

**Goal:** Deliver the approved general screening data and multi-task training workflow without installing Protenix.

**Architecture:** Extend the current manifest and CLI; reuse native tensor preparation and Torch training. Keep public acquisition/import in `sources.py`, standard-library policy/splits in `data.py`, and task output routing in `training.py`.

**Tech Stack:** Python standard library; existing optional PyTorch; native Protenix only on the user's training server.

**Spec:** [approved design](../specs/2026-09-17-general-screening-design.md).

## Constraints

Preserve legacy inputs and checkpoints; archive superseded documents. No implicit install or download. No failed/unknown assay becomes a negative. No cell phenotype becomes a protein-binding label. Both-cold splits retain excluded rows for auditing. All generated artifacts refuse overwrite.

## Tasks

- [x] Add failing integration tests in `tests/test_general.py`: public CSV outcome/QC mapping, LIT import, cold splits and leakage, assay-separated metrics, task-specific gradients and transfer.
- [x] Extend `fragment_ft/data.py`: optional observation fields, eligibility and task identity, three split strategies, coverage audit and per-assay metrics.
- [x] Add `fragment_ft/sources.py`: explicit downloads with hash receipts, PubChem AID acquisition, reviewed mapping and compound joins, LIT import.
- [x] Extend `training.py`, `protenix_adapter.py`, `synthetic.py`, `__main__.py`: task routing, balanced sampling, transfer initialization, prediction metadata, X-ray-only synthetic policy.
- [x] Add runnable multi-target example and mapping configurations. Rewrite README/data/training guidance; archive the original two-target plan.
- [x] Run `python3 -m unittest discover -s tests -v`, existing-Torch tests, CLI example chain, syntax checks, live small public download, focused independent review; resolve findings.

The prior discussion and explicit implementation request supply design approval. Execution is inline; the folder is not a Git checkout, so no commits/worktrees are created.

Completed 2026-09-17. See [verification record](../../VALIDATION.md): 22 existing-Torch CPU tests pass; standard Python passes 13 and skips 9. Native Protenix/GPU execution remains outside this local verification.
