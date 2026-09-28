# Fragment fine-tuning implementation plan

**Goal:** Build the program described in FINETUNING_PLAN.md without installing Protenix or downloading model weights.

**Architecture:** A standard-library CLI validates assay manifests and splits shared chemistry groups across both targets. An optional Protenix adapter prepares native tensors and connects its existing trunk/structure loss to a small binding head. A PyTorch loop trains the head alone or jointly with explicitly selected backbone parameters. The default structural branch uses positive complexes; the user-approved synthetic ablation adds separately sampled artificial negatives from observed apo structures.

**Tech stack:** Python 3.11+, standard library, existing PyTorch for numerical tests. Protenix and its dependencies are required only on the eventual training machine. No installation is performed by this program.

**Spec:** FINETUNING_PLAN.md and the user's instruction to implement without installing Protenix.

## Constraints

- Shared fragment/chemistry groups stay in the same split across targets.
- Unknown and failed assays never become negative labels.
- Binding inputs come from target-level protein contexts plus SMILES, never positive complex coordinates.
- Default structural batches come only from positive experimental complexes. Artificial negative labels require explicit opt-in and remain marked synthetic.
- Native integration targets upstream commit `4c355be4553512f72453ecbfb65e69f4c35d1413`.
- No implicit package, checkpoint, MSA or database downloads.
- Preserve existing files; refuse overwriting prepared artifacts/checkpoints.

## Tasks

- [x] Write workflow tests: leakage rejection, deterministic group split, uncertain exclusion, negative structure exclusion, gradients and checkpoint round-trip.
- [x] Implement manifest validation, split assignment, upstream input export, and per-target classification metrics.
- [x] Implement lazy native preparation and Protenix adapter using inspected API signatures.
- [x] Implement head/joint training, validation, checkpoint resume, prediction and native weight export.
- [x] Add explicit synthetic ablation, integral one-to-one apo mapping and reproducible artificial placement.
- [x] Add examples and documented commands, including torchrun usage and the boundary between tested local logic and unexecuted native integration.
- [x] Run standard-library tests, existing-PyTorch CPU tests, CLI examples, syntax checks and a final code review.

Verification: 13 tests pass in the existing PyTorch CPU environment; standard-library Python passes 7 and skips 6 optional PyTorch checks. CLI help/validate/split/inputs and clean failure without Protenix pass. Native checkpoint, real apo/CIF, CUDA/BF16 and distributed execution remain untested and require the user's eventual training environment and data.
