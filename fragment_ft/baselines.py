"""Train-only screening baselines and imported, uncalibrated Protenix scores."""
import argparse
import math
from pathlib import Path
import sys

from .data import (file_hash, predictable, prediction_report, read_json, read_manifest,
                   row_hash, task_name, training_rows, validate_rows, write_json)


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return value


def register_parser(commands):
    """Register one command on the parent's argparse subparser collection."""
    command = commands.add_parser('baseline', help='Train-only baselines or external Protenix confidence scores')
    command.add_argument('manifest')
    command.add_argument('--method', choices=['prior', 'ligand', 'protenix'], required=True)
    command.add_argument('--output', required=True)
    command.add_argument('--split', choices=['train', 'val', 'test', 'all'], default='test')
    command.add_argument('--top-k', type=positive_int, default=20)
    command.add_argument('--scores', help='Reviewed external score JSON; protenix only')
    command.add_argument('--metric', choices=['iptm', 'pae', 'plddt'])
    command.add_argument('--orientation', choices=['higher', 'lower'])
    command.add_argument('--score-scale', type=float,
                         help='iptm: 1; plddt: 1 or 100; pae: positive Angstrom scale')
    return command


def fit_prior(train, selected):
    """Target/task hit rate, falling back to the train-only task rate for cold targets."""
    rates, target_rates = {}, {}
    for task in sorted({task_name(row) for row in selected}):
        examples = [row for row in train if task_name(row) == task]
        if not examples:
            raise ValueError(f'No reliable train observations for task {task}')
        rates[task] = sum(int(row['label']) for row in examples) / len(examples)
        target_rates[task] = {}
        for target in sorted({row['target_id'] for row in examples}):
            target_rows = [row for row in examples if row['target_id'] == target]
            target_rates[task][target] = sum(int(row['label']) for row in target_rows) / len(target_rows)
    probabilities = [target_rates[task_name(row)].get(row['target_id'], rates[task_name(row)])
                     for row in selected]
    return probabilities, {'task_hit_rates': rates, 'target_hit_rates': target_rates,
                           'fallback_policy': 'unseen target/task uses reliable train task-global hit rate',
                           'fallback_sample_ids': [row['sample_id'] for row in selected
                                                   if row['target_id'] not in target_rates[task_name(row)]]}


def fit_ligand(train, selected):
    """Fixed descriptor logistic regression per task; all learned state is train-only."""
    try:
        from rdkit import Chem, rdBase
        from rdkit.Chem import Descriptors, Lipinski
    except ImportError:
        raise ImportError('The ligand baseline requires an existing RDKit installation') from None
    descriptors = (Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA,
                   Lipinski.NumHDonors, Lipinski.NumHAcceptors, Lipinski.NumRotatableBonds)
    cache = {}

    def features(row):
        smiles = row['smiles']
        if smiles not in cache:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise ValueError(f'{row["sample_id"]}: RDKit cannot parse SMILES {smiles!r}')
            values = [float(descriptor(molecule)) for descriptor in descriptors]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f'{row["sample_id"]}: nonfinite ligand descriptor')
            cache[smiles] = values
        return cache[smiles]

    def sigmoid(value):
        return 1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value))

    models = {}
    for task in sorted({task_name(row) for row in selected}):
        examples = sorted((row for row in train if task_name(row) == task), key=lambda row: row['sample_id'])
        if not examples:
            raise ValueError(f'No reliable train observations for task {task}')
        labels = [int(row['label']) for row in examples]
        if len(set(labels)) != 2:
            raise ValueError(f'Ligand baseline requires both train classes for task {task}; use prior instead')
        matrix = [features(row) for row in examples]
        means = [sum(column) / len(matrix) for column in zip(*matrix)]
        scales = [math.sqrt(sum((row[j] - mean)**2 for row in matrix) / len(matrix)) or 1.0
                  for j, mean in enumerate(means)]
        matrix = [[1.0] + [(value - mean) / scale for value, mean, scale in zip(row, means, scales)]
                  for row in matrix]
        weights = [0.0] * 7
        # Fixed full-batch schedule, no random seed, validation selection or held-out fitting.
        for _ in range(500):
            errors = [sigmoid(sum(w*x for w, x in zip(weights, row))) - label
                      for row, label in zip(matrix, labels)]
            gradient = [sum(error * row[j] for error, row in zip(errors, matrix)) / len(matrix)
                        + (0.01 * weight if j else 0.0) for j, weight in enumerate(weights)]
            weights = [weight - 0.1 * grad for weight, grad in zip(weights, gradient)]
        models[task] = {'means': means, 'scales': scales, 'weights': weights}
    probabilities = []
    for row in selected:
        model = models[task_name(row)]
        values = [1.0] + [(value - mean) / scale for value, mean, scale in
                          zip(features(row), model['means'], model['scales'])]
        probabilities.append(sigmoid(sum(w*x for w, x in zip(model['weights'], values))))
    return probabilities, {'rdkit_version': rdBase.rdkitVersion, 'models': models,
                           'descriptors': ['MolWt', 'MolLogP', 'TPSA', 'NumHDonors',
                                           'NumHAcceptors', 'NumRotatableBonds'],
                           'steps': 500, 'learning_rate': 0.1, 'l2': 0.01}


def import_scores(args, selected):
    """Import exactly one reviewed scalar per selected sample, without fitting."""
    if not args.scores or not args.metric or not args.orientation or args.score_scale is None:
        raise ValueError('protenix requires --scores, --metric, --orientation and --score-scale')
    scale = args.score_scale
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('--score-scale must be positive and finite')
    expected_direction = 'lower' if args.metric == 'pae' else 'higher'
    if args.orientation != expected_direction:
        raise ValueError(f'{args.metric} requires --orientation {expected_direction}')
    if args.metric == 'iptm' and scale != 1 or args.metric == 'plddt' and scale not in (1, 100):
        raise ValueError('iptm scale must be 1; plddt scale must be 1 or 100')
    payload = read_json(args.scores)
    if not isinstance(payload, dict) or set(payload) != {'metric', 'provenance', 'scores'}:
        raise ValueError('Score JSON requires exactly metric, provenance and scores')
    if payload['metric'] != args.metric:
        raise ValueError('Score JSON metric differs from --metric')
    provenance = payload['provenance']
    required = {'source', 'model', 'score_definition', 'selection_policy'}
    if (not isinstance(provenance, dict) or not required <= set(provenance)
            or not all(isinstance(value, str) and value.strip() for value in provenance.values())):
        raise ValueError('provenance requires nonempty strings: source, model, score_definition, selection_policy')
    records = payload['scores']
    if not isinstance(records, list):
        raise ValueError('scores must be a list')
    expected = {row['sample_id']: row for row in selected}
    imported = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {'sample_id', 'target_id', 'ligand_id', 'score'}:
            raise ValueError('Each score requires exactly sample_id, target_id, ligand_id, score')
        if not all(isinstance(record[key], str) for key in ('sample_id', 'target_id', 'ligand_id')):
            raise ValueError('Score identities must be strings')
        sid = record['sample_id']
        if sid in imported:
            raise ValueError(f'Duplicate score sample_id: {sid}')
        if sid not in expected:
            raise ValueError(f'Unexpected score sample_id: {sid}')
        if any(record[key] != expected[sid][key] for key in ('target_id', 'ligand_id')):
            raise ValueError(f'{sid}: score target_id or ligand_id mismatch')
        value = record['score']
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'{sid}: score must be a finite JSON number')
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite or value < 0 or (args.metric != 'pae' and value > scale):
            raise ValueError(f'{sid}: score outside finite {args.metric} range')
        imported[sid] = value
    missing = sorted(set(expected) - set(imported))
    if missing:
        raise ValueError(f'Missing scores for selected samples: {missing}')
    raw = [imported[row['sample_id']] for row in selected]
    probabilities = [1 / (1 + value / scale) if args.metric == 'pae' else value / scale for value in raw]
    return probabilities, {'metric': args.metric, 'orientation': args.orientation, 'scale': scale,
                           'transform': '1/(1+score/scale)' if args.metric == 'pae' else 'score/scale',
                           'scores_sha256': file_hash(args.scores), 'provenance': provenance,
                           'raw_scores': records}


def run(args):
    """Write a compare-compatible report; parent CLI handles exceptions and exit codes."""
    if Path(args.output).exists():
        raise FileExistsError(f'Refusing to overwrite report: {args.output}')
    rows = read_manifest(args.manifest)
    validate_rows(rows, require_splits=True)
    selected = [row for row in rows if predictable(row) and (args.split == 'all' or row['split'] == args.split)]
    if not selected:
        raise ValueError('No predictable observations in selected split')
    manifest_hash = file_hash(args.manifest)
    train, _ = training_rows(rows)
    if args.method == 'protenix':
        probabilities, details = import_scores(args, selected)
        training = {'method': 'protenix', 'fit_sample_ids': [], 'fit_row_hashes': []}
    else:
        if any(getattr(args, key) is not None for key in ('scores', 'metric', 'orientation', 'score_scale')):
            raise ValueError('Score import flags are valid only for --method protenix')
        tasks = {task_name(row) for row in selected}
        train = sorted((row for row in train if task_name(row) in tasks), key=lambda row: row['sample_id'])
        probabilities, details = (fit_prior if args.method == 'prior' else fit_ligand)(train, selected)
        training = {'method': args.method, 'manifest_sha256': manifest_hash,
                    'fit_sample_ids': [row['sample_id'] for row in train],
                    'fit_row_hashes': [row_hash(row) for row in train],
                    'weight_policy': 'unweighted reliable observations; weight > 0 is eligibility only'}
    predictions = [dict(row, task=task_name(row), probability=value) for row, value in zip(selected, probabilities)]
    report = {'baseline': args.method, 'manifest_sha256': manifest_hash, 'split': args.split,
              'training': training, 'same_manifest_as_training': True if args.method != 'protenix' else None,
              'score_semantics': 'uncalibrated confidence proxy' if args.method == 'protenix'
              else 'train-only empirical probability estimate; no held-out calibration',
              'details': details, 'predictions': predictions,
              'metrics_by_target': prediction_report(predictions, args.top_k)}
    if args.method == 'protenix':
        report['raw_scores'] = details.pop('raw_scores')
        report['provenance'] = details.pop('provenance')
    write_json(args.output, report)
    return 0


def main(argv=None):
    """Standalone CLI until the parent registers and dispatches baseline."""
    root = argparse.ArgumentParser(description=__doc__)
    register_parser(root.add_subparsers(dest='command', required=True))
    args = root.parse_args(argv)
    try:
        return run(args)
    except (ValueError, OSError, ImportError) as error:
        print(f'error: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
