# P0-2 baseline ladder

These baselines test whether a trained binding head improves on prevalence,
ligand chemistry alone, and zero-shot Protenix confidence. They do not run
Protenix, install packages, download weights, or synthesize native measurements.

## Commands

Use the main CLI:

```bash
python3 -m fragment_ft baseline examples/manifest.csv --method prior --split test --output private/prior.json
python3 -m fragment_ft baseline examples/manifest.csv --method ligand --split test --output private/ligand.json
python3 -m fragment_ft baseline examples/manifest.csv --method protenix --split test --scores private/iptm.json --metric iptm --orientation higher --score-scale 1 --output private/zero-shot.json
python3 -m fragment_ft compare private/prior.json private/ligand.json private/zero-shot.json --output private/comparison.json
```

Create the output parent directory first. The manifest must already have valid
splits; use your reviewed, split manifest instead of the fictional example for
scientific comparisons. Outputs refuse overwrite. `--split` accepts `train`,
`val`, `test` (default), or `all`; `--top-k` is a positive integer (default 20).
All reports being compared must select exactly the same manifest observations.

The main CLI returns 0 after writing the report and 2 for invalid input,
unavailable dependencies, or filesystem errors.

## Training and reliability policy

Both fitted baselines use only `split=train` rows accepted by
`data.training_rows`: binary labels, passing quality, positive weight,
non-phenotypic, and non-excluded. Weight is an eligibility gate, not a loss
multiplier. No validation/test label, descriptor statistic, optimization step,
hyperparameter selection, or calibration is used for fitting. Models are
separate for each `assay_type:endpoint` task. There is
no fallback across endpoints: a selected task without reliable training data
is an error.

* **prior:** unweighted positive fraction among reliable train observations of
  the same target and task. A target/task with no reliable training observations
  uses the task-global reliable training fraction across targets. Zero or one
  prevalence is allowed. `details.target_hit_rates` is keyed by task then target;
  `details.task_hit_rates` records the fallback rates. `fallback_policy` and
  `fallback_sample_ids` explicitly identify the rule and affected predictions.
* **ligand:** requires an already installed RDKit, but not NumPy or Torch.
  Features are MolWt, MolLogP, TPSA, NumHDonors, NumHAcceptors, and
  NumRotatableBonds. Means and population standard deviations are fitted on
  reliable train rows only; constant descriptors use scale 1. A logistic model
  per task uses an intercept, zero initialization, 500 full-batch gradient steps,
  learning rate 0.1, and L2 coefficient 0.01 on non-intercept weights. Training
  rows are sorted by sample ID. No random state or early stopping is used.
  Both classes are required for each selected task; otherwise use `prior`.
  This is a small chemistry-confounding control, not a tuned QSAR benchmark.

Predictions retain unknown, uncertain, failed-quality, and zero-weight
observations if predictable and in the selected split, just as `predict` does.
Their labels are never recoded as negatives. `prediction_report` and `compare`
score only reliable observations. Phenotypic and excluded rows are never
selected. Split leakage is rejected by the shared manifest validator. This is
the existing declared chemistry/target-group policy, not a new molecular
identity or external pretraining-overlap audit.

## External score schema

The input is UTF-8 JSON, with exactly `metric`, `provenance`, and `scores` at the
root. This example illustrates the schema only; numbers are fictional:

```json
{
  "metric": "iptm",
  "provenance": {
    "source": "path or URI of the original native output",
    "model": "exact model/checkpoint identity and version",
    "score_definition": "receptor-ligand interface ipTM from the specified chains",
    "selection_policy": "one predeclared seed and sample; no label-based selection"
  },
  "scores": [
    {"sample_id": "example_1", "target_id": "target_1", "ligand_id": "ligand_1", "score": 0.72}
  ]
}
```

`metric` is exactly `iptm`, `pae`, or `plddt` and must match `--metric`.
Provenance requires all four shown keys; additional string-valued keys may
record commit, seed, chain IDs, checkpoint hash, or other context. Every value
must be a nonempty string. Each score record has exactly the four shown keys.
Identity fields are strings, matched case-sensitively against the manifest.
There must be exactly one record for **every selected predictable sample**,
including samples ineligible for metrics. Missing, duplicate, extra,
wrong-target, and wrong-ligand records are errors, even if an extra record
belongs to another split in the same manifest. Order is irrelevant. Booleans,
numeric strings, nonfinite numbers, negative scores, and duplicate JSON keys
are rejected.

| Metric | Required orientation | Explicit scale | Raw range | Stored probability field |
| --- | --- | --- | --- | --- |
| iptm | `higher` | `1` | [0, 1] | score |
| plddt | `higher` | `1` or `100` | [0, scale] | score / scale |
| pae | `lower` | finite positive Angstrom value | [0, infinity), finite | 1 / (1 + score / scale) |

For example, PAE uses `--metric pae --orientation lower --score-scale 10`;
pLDDT in the usual 0-100 units uses `--metric plddt --orientation higher
--score-scale 100`. Choose the PAE scale before looking at held-out labels; it
is not estimated from scores or labels. The importer neither infers units nor
clips invalid confidence values. It accepts a reviewed scalar, not native
arrays: record exactly which chains, interface, residues, and native field
produced it in `score_definition`. Select one native output by a predeclared
policy outside this importer. No seed pooling, structure aggregation, or
held-out selection is implemented.

## Report schema and interpretation

The report root contains `baseline`, `manifest_sha256`, `split`, `training`,
`same_manifest_as_training`, `score_semantics`, `details`, `predictions`, and
`metrics_by_target`. Imported reports additionally contain root `raw_scores`
and `provenance`. `raw_scores` preserves the input score records and ordering;
`provenance` preserves the reviewed provenance object. `details` records
metric, orientation, scale, transform, and the SHA256 of the import file, or
the prior rates / learned ligand model, schedule, features, and RDKit version.

`training` records method, exact fit sample IDs, fit row hashes, and, for fitted
models, manifest SHA256 and weight policy. Zero-shot reports have empty fit
lists and `same_manifest_as_training: null`; they do not claim absence of
overlap with upstream pretraining. Fitted models record `true` because they
fit the train partition of this same manifest.

Each prediction is exactly the original normalized manifest row plus `task`
and numeric `probability`. No raw score or metadata is inserted into a
prediction: `compare` checks the manifest identity of all other fields.

**Protenix confidence is not a calibrated binding probability.** The
`probability` name is required by the existing comparison schema. These
monotonic transformations make larger values mean more favorable confidence;
they do not establish binding, affinity, or calibrated risk. Ranking metrics
can compare the fixed proxies with other baselines. Brier score and log loss
are mechanically emitted, but cannot be interpreted as calibration evidence
for these confidence proxies. Fitted baselines also have no held-out
calibration. Do not claim native Protenix execution from an import-only run.

## Verification

```bash
python3 -m unittest discover -s tests -p test_baselines.py -v
```

RDKit tests skip when it is unavailable; run with an existing RDKit interpreter
to exercise the ligand model. Tests cover train-only fitting, label-policy
exclusion, score direction/ranges, exact sample matching, schema rejection,
parser/runner integration, the standalone process, and real `compare` calls.
