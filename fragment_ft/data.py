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


def number(row, key):
    """Finite float from a manifest cell; errors name the row and column."""
    try:
        value = float(row[key])
    except (TypeError, ValueError):
        value = math.nan
    if not math.isfinite(value):
        raise ValueError(f'{row.get("sample_id", "")}: {key} must be a finite number, got {row[key]!r}')
    return value


def trainable(row):
    return (predictable(row) and row['label'] in ('0', '1') and (row.get('quality') or 'pass') == 'pass'
            and (number(row, 'weight') if row.get('weight') else 1) > 0)


def synthetic_eligible(row):
    return trainable(row) and row['label'] == '0' and (row.get('assay_type') or 'xray') == 'xray'


def structure_eligible(row):
    return row['label'] == '1' and bool(row.get('structure_id'))


def observation_key(row):
    """One measurement of one ligand on one target in one assay; missing fields count as ''."""
    return tuple(row.get(key, '') for key in ('source', 'assay_id', 'target_id', 'ligand_id'))


def packet_name(row, kind):
    return f'{row["sample_id"]}.{kind}.pt'


def compact(row):
    """FIELDS order; base fields always, observation fields only when nonempty; values stripped."""
    return {k: v for k in FIELDS if (v := str(row.get(k) or '').strip()) or k in BASE_FIELDS}


def training_rows(rows):
    """Reliable train and val observations; nothing else reaches the loss or checkpoint selection."""
    return ([r for r in rows if r['split'] == 'train' and trainable(r)],
            [r for r in rows if r['split'] == 'val' and trainable(r)])


def build_buckets(train):
    buckets = {}
    for row in train:
        target = buckets.setdefault(task_name(row), {}).setdefault(row['target_id'], {})
        target.setdefault((row.get('source', ''), row.get('assay_id', '')), []).append(row)
    return [[list(target.values()) for target in task.values()] for task in buckets.values()]


def sample_row(buckets, train, rng, sampling):
    # Select task, then target, then assay uniformly; large campaigns cannot dominate by row count.
    return rng.choice(rng.choice(rng.choice(rng.choice(buckets)))) if sampling == 'assay' else rng.choice(train)


def selection_scores(report):
    """Equal task weight, then equal weight for each target/assay group within a task.
    Average precision skips groups without positives; None when no group has one."""
    losses, precisions = defaultdict(list), defaultdict(list)
    for item in report.values():
        losses[item['task']].append(item['log_loss'])
        if item['average_precision'] is not None:
            precisions[item['task']].append(item['average_precision'])
    mean = lambda values: sum(values) / len(values)
    return {'average_precision': mean([mean(v) for v in precisions.values()]) if precisions else None,
            'log_loss': mean([mean(v) for v in losses.values()])}


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
        # A dropped 'Quality' or 'AssayType' column would silently turn rows into xray:hit QC-pass labels.
        unknown = [name for name in reader.fieldnames if name not in FIELDS]
        if unknown:
            renames = [f'rename {name!r} to {fixed!r}' for name in unknown
                       if (fixed := re.sub(r'[ -]', '_', name.strip().lower())) in FIELDS]
            raise ValueError(f'Unknown manifest columns: {unknown}' + ''.join(f'; {r}' for r in renames))
        missing = set(FIELDS[:6]) - set(reader.fieldnames)
        if missing:
            raise ValueError(f'Missing manifest columns: {sorted(missing)}')
        rows = []
        for row in reader:
            if None in row:
                raise ValueError('CSV contains extra columns; quote SMILES containing commas')
            rows.append(compact(row))
    validate_rows(rows)
    return rows


def validate_rows(rows, require_splits=False):
    if not rows:
        raise ValueError('Manifest is empty')
    strategy = split_strategy(rows)
    ids, ligands, smiles_groups, splits, structures, targets, assays = set(), {}, {}, {}, {}, {}, {}
    observed = defaultdict(set)
    for row in rows:
        sid = row.get('sample_id', '')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', sid):
            raise ValueError(f'Unsafe or missing sample_id: {sid!r}')
        # Packet files are named by sample_id; macOS and Windows filesystems ignore case.
        if sid.lower() in ids:
            raise ValueError(f'Duplicate sample_id (case-insensitive): {sid}')
        ids.add(sid.lower())
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
            if row.get('weight') and not 0 <= number(row, 'weight') <= 1:
                raise ValueError(f'{sid}: weight must be between 0 and 1')
            if row.get('concentration'):
                if number(row, 'concentration') <= 0 or row.get('concentration_unit') not in ('M', 'mM', 'uM', 'nM', 'pM'):
                    raise ValueError(f'{sid}: concentration requires a positive value and molar unit')
            if row.get('measurement_value'):
                number(row, 'measurement_value')
                if not row.get('measurement_unit'):
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
            raise ValueError(f'{sid}: split must be train, val, test or excluded' if split else
                             f'{sid}: split is missing; run the split command first')
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
        if trainable(row):
            observed[observation_key(row)].add(label)
    conflicts = sorted(key for key, labels in observed.items() if len(labels) > 1)
    if conflicts:
        raise ValueError(f'{len(conflicts)} observation(s) have both 0 and 1 labels for one '
                         f'(source, assay_id, target_id, ligand_id), e.g. {conflicts[0]}; review and mark one uncertain')
    return rows


def assign_splits(rows, seed=42, validation=0.15, test=0.15, strategy='chemistry'):
    """Fractions cap trainable rows per held-out split (a split no group fits gets the smallest one);
    groups are shuffled from sorted order, so outputs are deterministic."""
    validate_rows(rows)
    if any(r.get('split') for r in rows):
        raise ValueError('Manifest already has splits; preserve the original split assignment')
    if not (0 < validation < 1 and 0 < test < 1 and validation + test < 1):
        raise ValueError('Split fractions must be positive with validation + test < 1')
    if strategy not in ('chemistry', 'target', 'both'):
        raise ValueError('Unknown split strategy')
    if strategy != 'target':
        shared = defaultdict(set)
        for r in rows:
            if r.get('structure_id'):
                shared[r['structure_id']].add(r['chem_group'])
        for structure, groups in sorted(shared.items()):
            if len(groups) > 1:
                raise ValueError(f'structure_id {structure} spans chem_groups {sorted(groups)}; merge these chem_groups')
    def partition(field, field_seed):
        if any(not r.get(field) for r in rows):
            raise ValueError(f'{field} is required for {strategy} splitting')
        # Groups without trainable rows cannot fill val/test quotas; they stay in train on this axis.
        sizes = Counter(r[field] for r in rows if trainable(r))
        groups = sorted(sizes)
        if len(groups) < 3:
            raise ValueError(f'At least 3 {field} groups with trainable observations are needed')
        random.Random(field_seed).shuffle(groups)
        total, assignment = sum(sizes.values()), {}
        for name, fraction, keep in (('val', validation, 2), ('test', test, 1)):
            # Only groups that fit the quota, so one large group cannot swallow train; keep >= 1 group each for later splits.
            free, filled = [g for g in groups if g not in assignment][:-keep], 0
            for group in free:
                if filled + sizes[group] <= fraction * total:
                    assignment[group] = name
                    filled += sizes[group]
            if not filled:  # every group overshoots: take the smallest
                assignment[min(free, key=sizes.get)] = name
        return defaultdict(lambda: 'train', assignment)
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
    counts = Counter(r['split'] for r in result if trainable(r))
    empty = [name for name in ('train', 'val', 'test') if not counts[name]]
    if empty:
        raise ValueError(f'{strategy} split leaves no trainable observations in {empty}; '
                         'use chemistry or target for sparse target x chemistry matrices')
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


def read_json(path):
    """UTF-8 JSON; duplicate keys are rejected instead of silently keeping the last one."""
    def unique(pairs):
        duplicates = sorted(key for key, n in Counter(key for key, _ in pairs).items() if n > 1)
        if duplicates:
            raise ValueError(f'{path}: duplicate JSON keys {duplicates}')
        return dict(pairs)
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream, object_pairs_hook=unique)


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
    preparation = read_json(directory / 'preparation.json')
    artifacts = preparation.get('artifacts', {})
    for row in rows:
        kinds = ['binding']
        if row['split'] == 'train' and structure and structure_eligible(row):
            kinds.append('structure')
        if row['split'] == 'train' and synthetic and synthetic_eligible(row):
            kinds.append('synthetic')
        for kind in kinds:
            name = packet_name(row, kind)
            if name not in artifacts or file_hash(directory / name) != artifacts[name]:
                raise ValueError(f'Missing or changed prepared artifact: {name}')
    return preparation


def summary(rows):
    counts = Counter((r['target_id'], r.get('split') or 'unassigned', r['label']) for r in rows)
    repeats = Counter(observation_key(r) for r in rows if trainable(r))
    return {'samples': len(rows), 'ligands': len({r['ligand_id'] for r in rows}),
            'chemistry_groups': len({r['chem_group'] for r in rows}),
            'eligible_observations': sum(trainable(r) for r in rows),
            'replicate_observations': sum(n - 1 for n in repeats.values()),
            'tasks': dict(Counter(task_name(r) for r in rows if trainable(r))),
            'counts': [{'target': t, 'split': s, 'label': l, 'n': n}
                       for (t, s, l), n in sorted(counts.items())]}


def make_inputs(rows, targets):
    """One shared protein context per target; no assay label or holo structure input."""
    validate_rows(rows)
    inputs, families = [], defaultdict(set)
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
        if row.get('target_group'):
            families[sequence].add(row['target_group'])
        inputs.append({'name': row['sample_id'], 'sequences': [
            {'proteinChain': protein}, {'ligand': {'ligand': row['smiles'], 'count': 1}}]})
    # Target-cold splits trust target_group; one protein under two groups would sit on both sides.
    for groups in families.values():
        if len(groups) > 1:
            raise ValueError(f'Identical protein sequences span target_groups {sorted(groups)}; merge them')
    return inputs


def metrics(labels, probabilities, k=20):
    """Screening metrics; score ties are averaged over every order. @k values are None when n <= k."""
    if not labels or len(labels) != len(probabilities) or k < 1:
        raise ValueError('Metrics require equal nonempty lists and positive k')
    if any(y not in (0, 1) for y in labels) or any(
            isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1
            for p in probabilities):
        raise ValueError('Labels must be 0/1 and probabilities numbers in [0, 1]')
    ranked = sorted(zip(probabilities, labels), reverse=True)
    positives, n = sum(labels), len(labels)
    negatives = n - positives
    # Aggregate score ties, rather than making AP depend on arbitrary row order.
    groups = defaultdict(list)
    for score, label in ranked:
        groups[score].append(label)
    def top_hits(limit):  # expected positives among the top `limit` rows
        hits = seen = 0
        for group in groups.values():
            hits += sum(group) * min(max(0, limit - seen), len(group)) / len(group)
            seen += len(group)
        return hits
    tp = seen = negatives_seen = 0
    ap = pairs = 0.0
    for group in groups.values():
        gained = sum(group)
        seen += len(group); tp += gained
        if positives:
            ap += gained / positives * tp / seen
        # Mann-Whitney AUROC: a positive beats every lower-scored negative and half of the tied ones.
        pairs += gained * (negatives - negatives_seen - (len(group) - gained) / 2)
        negatives_seen += len(group) - gained
    top, fraction = top_hits(k), math.ceil(0.01 * n)
    eps = 1e-7
    return {'n': n, 'positives': positives, 'prevalence': positives / n,
            'average_precision': ap if positives else None,
            'auroc': pairs / (positives * negatives) if positives and negatives else None,
            'brier': sum((p-y)**2 for y, p in zip(labels, probabilities)) / n,
            'log_loss': -sum(y * math.log(max(eps, p)) + (1-y) * math.log(max(eps, 1-p))
                             for y, p in zip(labels, probabilities)) / n,
            'k': k, 'precision_at_k': top / k if n > k else None,
            'recall_at_k': top / positives if n > k and positives else None,
            'enrichment_at_k': top / k / (positives/n) if n > k and positives else None,
            'enrichment_at_1_percent': top_hits(fraction) / fraction / (positives/n)
            if fraction < n and positives else None}


def prediction_report(predictions, k=20):
    """Metrics per split, source, target, task and assay; only trainable observations are scored."""
    groups = defaultdict(list)
    for pred in predictions:
        if trainable(pred):
            key = [pred.get('split', ''), pred.get('source', ''), pred['target_id'], task_name(pred), pred.get('assay_id', '')]
            groups[json.dumps(key, separators=(',', ':'))].append(pred)
    return {key: metrics([int(r['label']) for r in records],
                         [r['probability'] for r in records], k=k) | {
                             'split': records[0].get('split', ''), 'task': task_name(records[0]),
                             'target_id': records[0]['target_id'], 'assay_id': records[0].get('assay_id', '')}
            for key, records in sorted(groups.items())}


def molecule_identity_check(rows, required=False):
    """One molecule (InChIKey) must keep one chem_group; validate_rows then keeps it in one split.
    Needs RDKit (a Protenix dependency); without it, returns checked=False unless required."""
    try:
        from rdkit import Chem
    except ImportError:
        if required:
            raise ImportError('RDKit is required to check molecule identity across SMILES spellings') from None
        return {'checked': False, 'molecules': 0}
    keys, groups = {}, defaultdict(set)
    for row in rows:
        if row['smiles'] not in keys:
            molecule = Chem.MolFromSmiles(row['smiles'])
            keys[row['smiles']] = Chem.MolToInchiKey(molecule) if molecule is not None else ''
        key = keys[row['smiles']]
        if not key:
            raise ValueError(f'{row["sample_id"]}: RDKit cannot parse SMILES {row["smiles"]!r}')
        groups[key].add(row['chem_group'])
    for key in sorted(groups):
        if len(groups[key]) > 1:
            raise ValueError(f'Molecule {key} spans chem_groups {sorted(groups[key])}; merge them')
    return {'checked': True, 'molecules': len(groups)}


def cross_audit(rows, reference_rows, limit=20):
    """Overlap of this manifest's val/test rows with a reference corpus's train rows (e.g. pretraining).
    leakage counts the fields this manifest's split strategy holds out, plus structure_id."""
    reference = [r for r in reference_rows if r.get('split') == 'train']
    if not reference:
        raise ValueError('Reference manifest has no train rows to audit against')
    held_out = [r for r in rows if r.get('split') in ('val', 'test')]
    strategy = split_strategy(rows)
    fields = {'structure_id'} | ({'smiles', 'ligand_id', 'chem_group'} if strategy != 'target' else set()) \
        | ({'target_group'} if strategy != 'chemistry' else set())
    overlap = {}
    for field in ('smiles', 'ligand_id', 'chem_group', 'target_group', 'structure_id'):
        shared = sorted({r.get(field) for r in held_out} & {r.get(field) for r in reference} - {'', None})
        overlap[field] = {'count': len(shared), 'examples': shared[:limit]}
    return {'strategy': strategy, 'held_out_rows': len(held_out), 'reference_train_rows': len(reference),
            'overlap': overlap, 'leakage': any(overlap[field]['count'] for field in fields)}
