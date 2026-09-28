"""Manifest, shared-library splits and evaluation; standard library only."""
from collections import Counter, defaultdict
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re

BASE_FIELDS = ('sample_id', 'target_id', 'ligand_id', 'smiles', 'chem_group', 'label',
               'split', 'structure_id', 'protein_chain_id', 'ligand_chain_id')
OBS_FIELDS = ('target_group', 'assay_id', 'assay_type', 'endpoint', 'campaign_id',
              'source', 'source_url', 'quality', 'weight', 'concentration', 'concentration_unit',
              'measurement_value', 'measurement_relation', 'measurement_unit', 'split_strategy')
FIELDS = BASE_FIELDS + OBS_FIELDS
LABELS = {'0', '1', 'unknown', 'uncertain'}
SPLITS = {'train', 'val', 'test', 'excluded'}
ASSAY_TYPES = {'xray', 'direct_binding', 'biochemical', 'phenotypic'}


def task_name(row):
    return f'{row.get("assay_type") or "xray"}:{row.get("endpoint") or "hit"}'


def predictable(row):
    return row.get('assay_type') != 'phenotypic' and row.get('split') != 'excluded'


def trainable(row):
    return (predictable(row) and row['label'] in ('0', '1')
            and (row.get('quality') or 'pass') == 'pass' and float(row.get('weight') or 1) > 0)


def synthetic_eligible(row):
    return trainable(row) and row['label'] == '0' and (row.get('assay_type') or 'xray') == 'xray'


def split_strategy(rows):
    strategies = {row.get('split_strategy') or 'chemistry' for row in rows}
    if len(strategies) != 1 or not strategies <= {'chemistry', 'target', 'both'}:
        raise ValueError('Use one split_strategy: chemistry, target or both')
    return next(iter(strategies))


def csv_reader(stream):
    """Reject ambiguous headers before DictReader can silently discard columns."""
    reader = csv.DictReader(stream, strict=True)
    headers = reader.fieldnames or []
    if not headers or any(not name.strip() for name in headers) or len(set(headers)) != len(headers):
        raise ValueError('CSV requires nonempty, unique column headers')
    return reader


def read_manifest(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv_reader(stream)
        missing = set(FIELDS[:6]) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'Missing manifest columns: {sorted(missing)}')
        rows = []
        for row in reader:
            if None in row:
                raise ValueError('CSV contains extra columns; quote SMILES containing commas')
            rows.append({k: (row.get(k) or '').strip() for k in FIELDS
                         if k in BASE_FIELDS or (row.get(k) or '').strip()})
    validate_rows(rows)
    return rows


def validate_rows(rows, require_splits=False):
    if not rows:
        raise ValueError('Manifest is empty')
    strategy = split_strategy(rows)
    ids, ligands, smiles_groups, splits, structures, targets, assays = set(), {}, {}, {}, {}, {}, {}
    for row in rows:
        sid = row.get('sample_id', '')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', sid):
            raise ValueError(f'Unsafe or missing sample_id: {sid!r}')
        if sid in ids:
            raise ValueError(f'Duplicate sample_id: {sid}')
        ids.add(sid)
        for key in FIELDS[1:6]:
            if not row.get(key):
                raise ValueError(f'{sid}: missing {key}')
        label = row['label']
        if label not in LABELS:
            raise ValueError(f'{sid}: label must be 0, 1, uncertain or unknown')
        if row['smiles'].startswith(('FILE_', 'CCD_')):
            raise ValueError(f'{sid}: smiles must contain SMILES, not a coordinate file or CCD reference')
        extended = any(row.get(k) for k in OBS_FIELDS if k != 'split_strategy')
        if extended:
            for key in OBS_FIELDS[:8]:
                if not row.get(key):
                    raise ValueError(f'{sid}: observation metadata requires {key}')
            if row['assay_type'] not in ASSAY_TYPES or not re.fullmatch(r'[a-z][a-z0-9_]*', row['endpoint']):
                raise ValueError(f'{sid}: invalid assay_type or endpoint')
            if row['quality'] not in ('pass', 'fail', 'uncertain'):
                raise ValueError(f'{sid}: quality must be pass, fail or uncertain')
            weight = float(row.get('weight') or 1)
            if not math.isfinite(weight) or not 0 <= weight <= 1:
                raise ValueError(f'{sid}: weight must be between 0 and 1')
            if row.get('concentration'):
                concentration = float(row['concentration'])
                if not math.isfinite(concentration) or concentration <= 0 or row.get('concentration_unit') not in ('M', 'mM', 'uM', 'nM', 'pM'):
                    raise ValueError(f'{sid}: concentration requires a positive value and molar unit')
            if row.get('measurement_value'):
                if not math.isfinite(float(row['measurement_value'])) or not row.get('measurement_unit'):
                    raise ValueError(f'{sid}: measurement requires a finite value and unit')
                if row.get('measurement_relation') not in ('=', '<', '<=', '>', '>='):
                    raise ValueError(f'{sid}: measurement_relation is required; preserve censored values')
            assay_key = (row['source'], row['assay_id'], row['target_id'])
            identity = (task_name(row), row['campaign_id'])
            if assay_key in assays and assays[assay_key] != identity:
                raise ValueError(f'{sid}: one assay_id has inconsistent endpoint/campaign metadata')
            assays[assay_key] = identity
        if strategy != 'chemistry' and not row.get('target_group'):
            raise ValueError(f'{sid}: target_group is required for target/both splitting')
        if row.get('target_group'):
            if row['target_id'] in targets and targets[row['target_id']] != row['target_group']:
                raise ValueError(f'{sid}: one target_id has inconsistent target_group')
            targets[row['target_id']] = row['target_group']
        ligand = row['ligand_id']
        identity = (row['smiles'], row['chem_group'])
        if ligand in ligands and ligands[ligand] != identity:
            raise ValueError(f'{ligand}: inconsistent smiles or chem_group across targets')
        ligands[ligand] = identity
        if row['smiles'] in smiles_groups and smiles_groups[row['smiles']] != row['chem_group']:
            raise ValueError(f'{sid}: identical SMILES have different chem_group values')
        smiles_groups[row['smiles']] = row['chem_group']
        split = row.get('split', '')
        if split not in SPLITS and (require_splits or split):
            raise ValueError(f'{sid}: split must be train, val or test')
        axes = ('chem_group',) if strategy == 'chemistry' else ('target_group',) if strategy == 'target' else ('chem_group', 'target_group')
        if split and split != 'excluded':
            for axis in axes:
                group = (axis, row[axis])
                if group in splits and splits[group] != split:
                    raise ValueError(f'{group}: split leakage across fragments/targets')
                splits[group] = split
        if split == 'excluded' and strategy != 'both':
            raise ValueError('excluded rows are reserved for double-cold (both) splits')
        structure = row.get('structure_id', '')
        if structure:
            if label != '1':
                raise ValueError(f'{sid}: structure_id is allowed only for positive labels')
            if extended and (row['quality'] != 'pass' or row['assay_type'] not in ('xray', 'direct_binding')):
                raise ValueError(f'{sid}: structure supervision requires a reliable observed binding positive')
            if not row.get('protein_chain_id') or not row.get('ligand_chain_id'):
                raise ValueError(f'{sid}: structure needs protein_chain_id and ligand_chain_id')
            if structure in structures and structures[structure] != (row['target_id'], split):
                raise ValueError(f'{structure}: inconsistent target or split for one structure')
            structures[structure] = (row['target_id'], split)
    return rows


def assign_splits(rows, seed=42, validation=0.15, test=0.15, strategy='chemistry'):
    validate_rows(rows)
    if any(r.get('split') for r in rows):
        raise ValueError('Manifest already has splits; preserve the original split assignment')
    if not (0 < validation < 1 and 0 < test < 1 and validation + test < 1):
        raise ValueError('Split fractions must be positive with validation + test < 1')
    if strategy not in ('chemistry', 'target', 'both'):
        raise ValueError('Unknown split strategy')
    def partition(field, field_seed):
        if any(not r.get(field) for r in rows):
            raise ValueError(f'{field} is required for {strategy} splitting')
        groups = sorted({r[field] for r in rows})
        if len(groups) < 3:
            raise ValueError(f'At least 3 independent {field} groups are needed')
        random.Random(field_seed).shuffle(groups)
        n_val, n_test = max(1, int(len(groups) * validation)), max(1, int(len(groups) * test))
        if n_val + n_test >= len(groups):
            raise ValueError('Split fractions leave no training groups')
        return {g: 'val' if i < n_val else 'test' if i < n_val + n_test else 'train'
                for i, g in enumerate(groups)}
    chemistry = partition('chem_group', seed) if strategy != 'target' else None
    targets = partition('target_group', seed + 1) if strategy != 'chemistry' else None
    result = []
    for row in rows:
        a = chemistry[row['chem_group']] if chemistry else None
        b = targets[row['target_group']] if targets else None
        split = a if strategy == 'chemistry' else b if strategy == 'target' else a if a == b else 'excluded'
        updates = {'split': split}
        if strategy != 'chemistry' or row.get('assay_type'):
            updates['split_strategy'] = strategy
        result.append(dict(row, **updates))
    return validate_rows(result, require_splits=True)


def split_audit(rows):
    validate_rows(rows, require_splits=True)
    coverage = []
    for task in sorted({task_name(r) for r in rows if trainable(r)}):
        for name in ('train', 'val', 'test'):
            labels = Counter(r['label'] for r in rows if trainable(r) and
                             task_name(r) == task and r['split'] == name)
            coverage.append({'task': task, 'split': name, 'positive': labels['1'],
                             'negative': labels['0'], 'both_classes': bool(labels['1'] and labels['0'])})
    return {'strategy': split_strategy(rows), 'task_coverage': coverage, 'splits': {
        name: {'observations': sum(r['split'] == name for r in rows),
               'trainable': sum(r['split'] == name and trainable(r) for r in rows),
               'targets': len({r['target_id'] for r in rows if r['split'] == name}),
               'chemistry_groups': len({r['chem_group'] for r in rows if r['split'] == name})}
        for name in ('train', 'val', 'test', 'excluded')}}


def write_manifest(path, rows):
    with Path(path).open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(text)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def row_hash(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()


def binding_key(row):
    # Feature randomness must not depend on assay labels, split or holo metadata.
    return row_hash({key: row[key] for key in ('target_id', 'ligand_id', 'smiles')})


def verify_features(directory, rows, structure=False, synthetic=False):
    directory = Path(directory)
    with (directory / 'preparation.json').open() as stream:
        preparation = json.load(stream)
    artifacts = preparation.get('artifacts', {})
    for row in rows:
        kinds = ['binding']
        if row['split'] == 'train' and structure and row['label'] == '1' and row.get('structure_id'):
            kinds.append('structure')
        if row['split'] == 'train' and synthetic and synthetic_eligible(row):
            kinds.append('synthetic')
        for kind in kinds:
            name = f'{row["sample_id"]}.{kind}.pt'
            if name not in artifacts or file_hash(directory / name) != artifacts[name]:
                raise ValueError(f'Missing or changed prepared artifact: {name}')
    return preparation


def summary(rows):
    counts = Counter((r['target_id'], r.get('split') or 'unassigned', r['label']) for r in rows)
    return {'samples': len(rows), 'ligands': len({r['ligand_id'] for r in rows}),
            'chemistry_groups': len({r['chem_group'] for r in rows}),
            'eligible_observations': sum(trainable(r) for r in rows),
            'tasks': dict(Counter(task_name(r) for r in rows if trainable(r))),
            'counts': [{'target': t, 'split': s, 'label': l, 'n': n}
                       for (t, s, l), n in sorted(counts.items())]}


def make_inputs(rows, targets):
    """One shared protein context per target; no assay label or holo structure input."""
    validate_rows(rows)
    inputs = []
    for row in rows:
        if row['target_id'] not in targets:
            raise ValueError(f'Missing target definition: {row["target_id"]}')
        protein = copy.deepcopy(targets[row['target_id']])
        protein.pop('apo', None)  # Metadata used only for the opt-in synthetic labels.
        sequence = protein.get('sequence', '')
        if not sequence or set(sequence) - set('ACDEFGHIKLMNPQRSTVWYX'):
            raise ValueError(f'{row["target_id"]}: invalid protein sequence')
        protein.setdefault('count', 1)
        if not isinstance(protein['count'], int) or protein['count'] < 1:
            raise ValueError('protein count must be a positive integer')
        inputs.append({'name': row['sample_id'], 'sequences': [
            {'proteinChain': protein}, {'ligand': {'ligand': row['smiles'], 'count': 1}}]})
    return inputs


def metrics(labels, probabilities, k=20):
    if not labels or len(labels) != len(probabilities) or k < 1:
        raise ValueError('Metrics require equal nonempty lists and positive k')
    if any(y not in (0, 1) for y in labels) or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError('Invalid labels or probabilities')
    ranked = sorted(zip(probabilities, labels), reverse=True)
    positives, n = sum(labels), len(labels)
    tp = seen = 0
    ap = 0.0
    # Aggregate score ties, rather than making AP depend on arbitrary row order.
    groups = defaultdict(list)
    for score, label in ranked:
        groups[score].append(label)
    top_hits = 0.0
    limit = min(k, n)
    for group in groups.values():
        gained = sum(group)
        remaining = max(0, limit - seen)
        top_hits += gained * min(remaining, len(group)) / len(group)
        seen += len(group); tp += gained
        if positives:
            ap += gained / positives * tp / seen
    eps = 1e-7
    return {'n': n, 'positives': positives, 'prevalence': positives / n,
            'average_precision': ap if positives else None,
            'brier': sum((p-y)**2 for y, p in zip(labels, probabilities)) / n,
            'log_loss': -sum(y * math.log(max(eps, p)) + (1-y) * math.log(max(eps, 1-p))
                             for y, p in zip(labels, probabilities)) / n,
            'k': limit, 'precision_at_k': top_hits / limit,
            'recall_at_k': top_hits / positives if positives else None,
            'enrichment_at_k': top_hits / limit / (positives/n) if positives else None}


def prediction_report(predictions, k=20):
    groups = defaultdict(list)
    for pred in predictions:
        if trainable(pred):
            key = json.dumps([pred['source'], pred['target_id'], task_name(pred), pred['assay_id']], separators=(',', ':')) \
                if pred.get('assay_type') else pred['target_id']
            groups[key].append(pred)
    return {key: metrics([int(r['label']) for r in records],
                         [r['probability'] for r in records], k=k) | {
                             'task': task_name(records[0]), 'target_id': records[0]['target_id'],
                             'assay_id': records[0].get('assay_id', '')}
            for key, records in sorted(groups.items())}
