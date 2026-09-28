"""Manifest rules, split policy, metrics and stdlib training helpers; no ML dependencies."""
from collections import Counter
import importlib.util
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest import mock

from fragment_ft import data


def obs(target, i, **changes):
    row = dict(sample_id=f'{target}_{i}', target_id=target, target_group=target, ligand_id=f'L{i}',
               smiles='C' * (i + 1), chem_group=f'G{i}', label=str(i % 2), split='', structure_id='',
               protein_chain_id='', ligand_chain_id='', assay_id=f'{target}_screen', assay_type='xray',
               endpoint='hit', campaign_id='study', source='example', source_url='https://example.org/study',
               quality='pass')
    row.update(changes)
    return row


def grid(targets='ABCDEF', n=12):
    return [obs(t, i) for t in targets for i in range(n)]


def write_csv(path, header, lines):
    Path(path).write_text('\n'.join([','.join(header)] + [','.join(line) for line in lines]) + '\n')


class ManifestSchemaTests(unittest.TestCase):
    def test_unknown_or_miscased_headers_are_rejected_not_dropped(self):
        base = ['sample_id', 'target_id', 'ligand_id', 'smiles', 'chem_group', 'label']
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'm.csv'
            for extra, message in [('Quality', "rename 'Quality' to 'quality'"),
                                   ('Assay Type', "rename 'Assay Type' to 'assay_type'"),
                                   ('Split', "rename 'Split' to 'split'"),
                                   ('notes', r"Unknown manifest columns: \['notes'\]$")]:
                write_csv(path, base + [extra], [['s1', 'A', 'L1', 'CC', 'G1', '0', 'fail']])
                with self.assertRaisesRegex(ValueError, message):
                    data.read_manifest(path)

    def test_compact_strips_and_keeps_schema_order(self):
        row = data.compact({'label': ' 1 ', 'weight': '  ', 'quality': ' pass', 'sample_id': 's', 'extra': 'x'})
        self.assertEqual(list(row), list(data.BASE_FIELDS) + ['quality'])
        self.assertEqual((row['label'], row['quality'], row['split']), ('1', 'pass', ''))

    def test_read_json_rejects_duplicate_keys_and_reads_utf8(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'x.json'
            path.write_text('{"a": {"b": 1, "b": 2}}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, r"duplicate JSON keys \['b'\]"):
                data.read_json(path)
            data.write_json(Path(tmp) / 'k.json', {'경로': '표적'})
            self.assertEqual(data.read_json(Path(tmp) / 'k.json'), {'경로': '표적'})


class ValidationTests(unittest.TestCase):
    def reject(self, rows, pattern):
        with self.assertRaisesRegex(ValueError, pattern):
            data.validate_rows(rows)

    def test_bad_observation_metadata(self):
        cases = [({'weight': '1.5'}, 'weight must be between'),
                 ({'weight': '0,5'}, "A_0: weight must be a finite number, got '0,5'"),
                 ({'weight': 'nan'}, 'A_0: weight must be a finite number'),
                 ({'concentration': '10 mM', 'concentration_unit': 'mM'}, 'A_0: concentration must be a finite number'),
                 ({'concentration': 'inf', 'concentration_unit': 'mM'}, 'concentration must be a finite number'),
                 ({'measurement_value': '-inf', 'measurement_unit': 'uM', 'measurement_relation': '='},
                  'measurement_value must be a finite number'),
                 ({'concentration': '-1', 'concentration_unit': 'mM'}, 'positive value and molar unit'),
                 ({'concentration': '1', 'concentration_unit': 'mg/mL'}, 'positive value and molar unit'),
                 ({'measurement_value': 'abc', 'measurement_unit': 'uM', 'measurement_relation': '='}, 'measurement_value must be'),
                 ({'measurement_value': '5', 'measurement_relation': '='}, 'finite value and unit'),
                 ({'measurement_value': '5', 'measurement_unit': 'uM'}, 'measurement_relation is required'),
                 ({'assay_type': 'cell'}, 'assay_type or endpoint'),
                 ({'endpoint': 'Bad-End'}, 'assay_type or endpoint'),
                 ({'quality': 'ok'}, 'quality must be'),
                 ({'campaign_id': ''}, 'requires campaign_id'),
                 ({'label': '1', 'structure_id': 'S1', 'protein_chain_id': 'A', 'ligand_chain_id': 'B',
                   'assay_type': 'biochemical', 'endpoint': 'inhibition'}, 'reliable observed binding positive'),
                 ({'label': '1', 'structure_id': 'S1', 'protein_chain_id': 'A', 'ligand_chain_id': 'B',
                   'quality': 'fail'}, 'reliable observed binding positive'),
                 ({'label': '1', 'structure_id': 'S1'}, 'protein_chain_id and ligand_chain_id')]
        for changes, pattern in cases:
            with self.subTest(changes=changes):
                self.reject([obs('A', 0, **changes)] + grid('B', 2), pattern)
        self.reject([obs('A', 0), obs('A', 1, campaign_id='other')], 'inconsistent endpoint/campaign')
        self.reject([obs('A', 0, split='excluded'), obs('A', 1, split='train')], 'reserved for double-cold')

    def test_weight_zero_is_kept_but_not_trainable(self):
        row = obs('A', 0, weight='0')
        data.validate_rows([row])
        self.assertFalse(data.trainable(row))
        self.assertTrue(data.trainable(obs('A', 0, weight='0.5')))

    def test_sample_id_must_be_a_safe_case_insensitive_filename(self):
        for sid in ['../x', '/abs', '.hidden', 'a/b', '', 'a b']:
            with self.subTest(sid=sid):
                self.reject([obs('A', 0, sample_id=sid)], 'sample_id')
        for first, second in (('A_0', 'a_0'), ('a_0', 'A_0')):
            self.reject([obs('A', 0, sample_id=first), obs('A', 1, sample_id=second)], 'Duplicate sample_id')

    def test_identity_guards_that_prevent_hidden_leakage(self):
        self.reject([obs('A', 0), obs('A', 1, smiles='C', chem_group='G9')], 'identical SMILES')
        self.reject([obs('A', 0), obs('A', 1, target_group='Z')], 'inconsistent target_group')
        structure = dict(label='1', structure_id='1ABC', protein_chain_id='A', ligand_chain_id='B')
        self.reject([obs('A', 1, split='train', **structure), obs('A', 3, split='test', **structure)],
                    'inconsistent target or split')

    def test_conflicting_labels_rejected_and_replicates_counted(self):
        rows = [obs('A', 0), obs('A', 1, ligand_id='L0', smiles='C', chem_group='G0', label='1')]
        self.reject(rows, 'mark one uncertain')
        data.validate_rows([rows[0], dict(rows[1], label='uncertain')])
        data.validate_rows([rows[0], dict(rows[1], quality='fail')])
        data.validate_rows([rows[0], dict(rows[1], assay_id='other', campaign_id='other')])
        replicate = [rows[0], dict(rows[1], label='0')]
        self.assertEqual(data.summary(data.validate_rows(replicate))['replicate_observations'], 1)
        self.assertEqual(data.summary(grid('A', 4))['replicate_observations'], 0)


class SplitTests(unittest.TestCase):
    def test_golden_assignment(self):
        # Pins the published algorithm: any change here re-splits existing corpora.
        expected = {'chemistry': ('chem_group', ['G10', 'G6', 'G9'], ['G0', 'G3', 'G8']),
                    'target': ('target_group', ['B'], ['A'])}  # 12-row targets: only one fits an 18-row quota
        for strategy, (axis, val, test) in expected.items():
            rows = data.assign_splits(grid(), seed=4, validation=.25, test=.25, strategy=strategy)
            self.assertEqual([sorted({r[axis] for r in rows if r['split'] == s}) for s in ('val', 'test')], [val, test])
        both = data.assign_splits(grid(), seed=4, validation=.25, test=.25, strategy='both')
        self.assertEqual(Counter(r['split'] for r in both), {'excluded': 42, 'train': 24, 'val': 3, 'test': 3})

    def test_fractions_count_trainable_rows_and_ignore_untrainable_groups(self):
        rows = grid('AB', 10)  # ten chem groups of two rows: 15% of 20 trainable rows fits one group, of all 50 rows three
        rows += [obs('C', i, label='uncertain', ligand_id=f'U{i}', smiles=f'N{"C" * i}', chem_group='U')
                 for i in range(30)]
        for seed in range(10):
            result = data.assign_splits(rows, seed=seed)
            self.assertEqual({r['split'] for r in result if r['chem_group'] == 'U'}, {'train'})
            counts = Counter(r['split'] for r in result if data.trainable(r))
            self.assertEqual((counts['val'], counts['test']), (2, 2))
        rows = grid('ABCD', 4) + [obs('P', i, assay_type='phenotypic', endpoint='viability', assay_id='cell')
                                  for i in range(4)]
        for seed in range(10):
            result = data.assign_splits(rows, seed=seed, strategy='target')
            self.assertEqual({r['split'] for r in result if r['target_group'] == 'P'}, {'train'})

    def test_groups_that_overshoot_a_quota_stay_in_train(self):
        # One 40-row scaffold and nine of five: val/test take only groups that fit 15% of 85 rows.
        rows = [obs('A', i, chem_group='BIG' if i < 40 else f'G{i % 9}') for i in range(85)]
        for seed in range(20):
            result = data.assign_splits(rows, seed=seed)
            counts = Counter(r['split'] for r in result)
            self.assertEqual({r['split'] for r in result if r['chem_group'] == 'BIG'}, {'train'})
            self.assertEqual((counts['val'], counts['test']), (10, 10))
        # When no group fits, each held-out split takes one smallest group and train keeps the rest.
        rows = [obs('A', i, chem_group=f'G{min(i // 3, 3)}') for i in range(14)]  # sizes 3, 3, 3, 5
        for seed in range(10):
            result = data.assign_splits(rows, seed=seed, validation=.1, test=.1)
            groups = {s: {r['chem_group'] for r in result if r['split'] == s} for s in ('val', 'test')}
            self.assertEqual([len(g) for g in groups.values()], [1, 1])
            self.assertNotIn('G3', groups['val'] | groups['test'])

    def test_shared_structure_across_chem_groups_is_refused_before_splitting(self):
        structure = dict(label='1', structure_id='7ABC', protein_chain_id='A', ligand_chain_id='B')
        rows = grid('A', 6) + [obs('A', 7, **structure), obs('A', 9, **structure)]
        with self.assertRaisesRegex(ValueError, "7ABC spans chem_groups \\['G7', 'G9'\\]; merge"):
            data.assign_splits(rows, seed=0)
        merged = [dict(r, chem_group='G7') if r['chem_group'] == 'G9' else r for r in rows]
        data.assign_splits(merged, seed=0)

    def test_double_cold_refuses_empty_trainable_split(self):
        # Each target screens its own library, so chem groups nest inside targets.
        rows = [obs(t, i, ligand_id=f'{t}{i}', smiles='C' * (i + 1) + 'N' * (k + 1), chem_group=f'{t}{i // 4}')
                for k, t in enumerate('ABCDEF') for i in range(12)]
        with self.assertRaisesRegex(ValueError, r"no trainable observations in \['val', 'test'\]; use chemistry or target"):
            data.assign_splits(rows, seed=0, strategy='both')
        audit = data.split_audit(data.assign_splits(rows, seed=17, strategy='both'))
        self.assertTrue(all(audit['splits'][s]['trainable'] for s in ('train', 'val', 'test')))

    def test_resplit_is_refused(self):
        rows = data.assign_splits(grid(), seed=4, validation=.25, test=.25)
        with self.assertRaisesRegex(ValueError, 'already has splits'):
            data.assign_splits(rows, seed=5, validation=.25, test=.25)

    def test_double_cold_leakage_is_detected_on_both_axes(self):
        rows = data.assign_splits(grid(), seed=4, validation=.25, test=.25, strategy='both')
        # Chem group (G*) and target group (A-F) names do not collide, so one lookup serves both axes.
        split = {r[axis]: r['split'] for r in rows if r['split'] != 'excluded' for axis in ('chem_group', 'target_group')}
        for leaked, other in (('target_group', 'chem_group'), ('chem_group', 'target_group')):
            index = next(i for i, r in enumerate(rows) if split[r[other]] == 'train' and split[r[leaked]] == 'test')
            broken = [dict(r) for r in rows]
            broken[index]['split'] = 'train'
            with self.assertRaisesRegex(ValueError, f"'{leaked}'.*leakage"):
                data.validate_rows(broken, require_splits=True)

    def test_audit_reports_class_coverage(self):
        rows = data.assign_splits(grid(), seed=4, validation=.25, test=.25)
        coverage = data.split_audit(rows)['task_coverage']
        self.assertEqual([(c['split'], c['both_classes']) for c in coverage],
                         [('train', True), ('val', True), ('test', True)])
        rows = [dict(r, label='0') if r['split'] == 'test' else r for r in rows]
        self.assertFalse(data.split_audit(rows)['task_coverage'][2]['both_classes'])
        with self.assertRaisesRegex(ValueError, 'run the split command'):
            data.split_audit(grid())


class MetricsTests(unittest.TestCase):
    def test_top_k_ties_auroc_and_small_groups(self):
        tie = data.metrics([1, 0, 1, 0], [.5] * 4, k=2)
        self.assertEqual([tie[k] for k in ('precision_at_k', 'recall_at_k', 'enrichment_at_k', 'auroc')], [.5, .5, 1., .5])
        ordered = data.metrics([1, 0, 1, 0], [.9, .8, .7, .1], k=2)
        self.assertEqual((ordered['precision_at_k'], ordered['recall_at_k'], ordered['auroc']), (.5, .5, .75))
        self.assertAlmostEqual(data.metrics([1, 0, 1, 0], [.9, .5, .5, .1], k=2)['precision_at_k'], .75)
        small = data.metrics([1, 0, 0, 0], [.1, .9, .9, .9], k=4)
        self.assertEqual((small['k'], small['precision_at_k'], small['recall_at_k'], small['enrichment_at_k']),
                         (4, None, None, None))
        self.assertIsNone(data.metrics([0, 0], [.1, .2])['auroc'])

    def test_enrichment_at_one_percent(self):
        labels = [1, 1] + [0] * 198
        result = data.metrics(labels, [.9, .8] + [.1] * 198)
        self.assertAlmostEqual(result['enrichment_at_1_percent'], 100.)
        self.assertAlmostEqual(data.metrics(labels, [.5] * 200)['enrichment_at_1_percent'], 1.)
        self.assertIsNone(data.metrics([1], [.5])['enrichment_at_1_percent'])
        # n=150: the top ceil(1.5) = 2 rows, not int(1.5) = 1; one of the two positives ranks second.
        self.assertAlmostEqual(data.metrics([0, 1, 1] + [0] * 147, [.9, .8, .1] + [.2] * 147)['enrichment_at_1_percent'], 37.5)

    def test_probabilities_must_be_numbers(self):
        for bad in ['0.2', True, None, float('nan'), 1.5]:
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, 'probabilities'):
                data.metrics([1, 0], [bad, .1])

    def test_report_groups_by_split_and_assay_and_scores_only_trainable_rows(self):
        rows = [dict(obs('A', i, split=split, assay_id=assay), probability=.9 if i % 2 else .1)
                for split in ('train', 'test') for assay in ('A_screen', 'A_other') for i in range(4)]
        rows += [dict(obs('A', 8, split='test', quality='fail'), probability=.0),
                 dict(obs('A', 9, split='test', label='uncertain'), probability=.0)]
        report = data.prediction_report(rows)
        self.assertEqual(sorted((v['split'], v['assay_id'], v['n']) for v in report.values()),
                         [('test', 'A_other', 4), ('test', 'A_screen', 4), ('train', 'A_other', 4), ('train', 'A_screen', 4)])


class TrainingHelperTests(unittest.TestCase):
    def test_training_rows_are_exact(self):
        rows = [obs('A', 0, split='train'), obs('A', 1, split='train'), obs('A', 2, split='val'),
                obs('A', 3, split='test'), obs('A', 4, split='train', quality='fail'),
                obs('A', 5, split='val', label='uncertain'), obs('A', 6, split='train', weight='0'),
                obs('A', 7, split='val', assay_type='phenotypic', endpoint='viability', assay_id='cell')]
        train, validation = data.training_rows(rows)
        self.assertEqual([r['sample_id'] for r in train], ['A_0', 'A_1'])
        self.assertEqual([r['sample_id'] for r in validation], ['A_2'])

    def test_assay_sampling_balances_task_target_and_assay(self):
        train = [obs('A', i, sample_id=f'x{i}') for i in range(1000)]
        train += [obs(t, i, sample_id=f'{t}{i}', assay_type='biochemical', endpoint='inhibition')
                  for t in 'BC' for i in range(10)]
        buckets, rng = data.build_buckets(train), random.Random(0)
        draws = Counter(data.sample_row(buckets, train, rng, 'assay')['target_id'] for _ in range(4000))
        for target, share in {'A': .5, 'B': .25, 'C': .25}.items():
            self.assertAlmostEqual(draws[target] / 4000, share, delta=.03)
        uniform = Counter(data.sample_row(buckets, train, rng, 'uniform')['target_id'] for _ in range(4000))
        self.assertGreater(uniform['A'] / 4000, .95)

    def test_selection_scores_are_task_macro(self):
        report = {'a': {'task': 'x', 'log_loss': .2, 'average_precision': .5},
                  'b': {'task': 'x', 'log_loss': .4, 'average_precision': None},
                  'c': {'task': 'y', 'log_loss': .6, 'average_precision': 1.}}
        scores = data.selection_scores(report)
        self.assertAlmostEqual(scores['log_loss'], .45)
        self.assertAlmostEqual(scores['average_precision'], .75)
        self.assertIsNone(data.selection_scores({'b': report['b']})['average_precision'])

    def test_verify_features_requires_structure_packets_for_joint_training(self):
        row = obs('A', 1, split='train', label='1', structure_id='1ABC', protein_chain_id='A', ligand_chain_id='B')
        self.assertTrue(data.structure_eligible(row))
        self.assertFalse(data.structure_eligible(obs('A', 0)))
        with tempfile.TemporaryDirectory() as tmp:
            packet = Path(tmp) / data.packet_name(row, 'binding')
            packet.write_bytes(b'features')
            data.write_json(Path(tmp) / 'preparation.json', {'artifacts': {packet.name: data.file_hash(packet)}})
            data.verify_features(tmp, [row])
            with self.assertRaisesRegex(ValueError, r'A_1\.structure\.pt'):
                data.verify_features(tmp, [row], structure=True)


class IdentityTests(unittest.TestCase):
    def test_identical_sequences_must_share_target_group(self):
        rows = [obs('A', 0), obs('B', 1)]
        targets = {'A': {'sequence': 'ACD'}, 'B': {'sequence': 'ACD'}}
        with self.assertRaisesRegex(ValueError, r"target_groups \['A', 'B'\]"):
            data.make_inputs(rows, targets)
        data.make_inputs([dict(r, target_group='fam') for r in rows], targets)

    def test_molecule_identity_uses_inchikey_when_rdkit_exists(self):
        rows = [obs('A', 0), obs('A', 1, smiles='c1ccccc1', chem_group='G1'), obs('A', 2, smiles='C1=CC=CC=C1')]
        with mock.patch.dict(sys.modules, {'rdkit': None}):
            self.assertEqual(data.molecule_identity_check(rows), {'checked': False, 'molecules': 0})
            with self.assertRaisesRegex(ImportError, 'RDKit'):
                data.molecule_identity_check(rows, required=True)
        canonical = {'c1ccccc1': 'BENZENE', 'C1=CC=CC=C1': 'BENZENE', 'C': 'METHANE'}
        chem = types.SimpleNamespace(MolFromSmiles=lambda s: s if s in canonical else None,
                                     MolToInchiKey=canonical.get)
        with mock.patch.dict(sys.modules, {'rdkit': types.SimpleNamespace(Chem=chem), 'rdkit.Chem': chem}):
            with self.assertRaisesRegex(ValueError, r"BENZENE spans chem_groups \['G1', 'G2'\]"):
                data.molecule_identity_check(rows)
            same = [rows[0], rows[1], dict(rows[2], chem_group='G1')]
            self.assertEqual(data.molecule_identity_check(same), {'checked': True, 'molecules': 2})
            with self.assertRaisesRegex(ValueError, 'cannot parse'):
                data.molecule_identity_check([obs('A', 3)], required=True)

    @unittest.skipUnless(importlib.util.find_spec('rdkit'), 'RDKit is not installed')
    def test_molecule_identity_with_real_rdkit(self):
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')  # the unparsable SMILES below would log to stderr
        self.addCleanup(RDLogger.EnableLog, 'rdApp.*')
        # PubChem writes Kekule SMILES, most libraries aromatic ones: one molecule, two spellings.
        kekule = obs('A', 0, smiles='CC(=O)OC1=CC=CC=C1C(=O)O', chem_group='Gpub')
        aromatic = obs('A', 1, smiles='CC(=O)Oc1ccccc1C(=O)O', chem_group='Gpriv')
        with self.assertRaisesRegex(ValueError, r"spans chem_groups \['Gpriv', 'Gpub'\]"):
            data.molecule_identity_check([kekule, aromatic])
        self.assertEqual(data.molecule_identity_check([kekule, dict(aromatic, chem_group='Gpub')], required=True),
                         {'checked': True, 'molecules': 1})
        with self.assertRaisesRegex(ValueError, 'cannot parse'):
            data.molecule_identity_check([obs('A', 2, smiles='C1CC')])

    def test_cross_audit_reports_held_out_overlap_with_reference_training(self):
        mine = [obs('A', 0, split='val'), obs('A', 1, split='train'), obs('B', 2, split='test')]
        reference = [obs('Z', 0, split='train', target_group='Z'), obs('Z', 1, split='test', target_group='Z')]
        audit = data.cross_audit(mine, reference)
        self.assertTrue(audit['leakage'])
        self.assertEqual(audit['overlap']['smiles'], {'count': 1, 'examples': ['C']})
        self.assertEqual(audit['overlap']['target_group']['count'], 0)
        clean = data.cross_audit(mine, [obs('A', 5, split='train')])
        self.assertEqual(clean['overlap']['target_group'], {'count': 1, 'examples': ['A']})
        self.assertFalse(clean['leakage'])  # a shared target is expected under a chemistry-cold split
        # structure_id counts toward leakage under every strategy, even target-cold where ligands may repeat.
        holo = dict(label='1', structure_id='1ABC', protein_chain_id='A', ligand_chain_id='B', split_strategy='target')
        target_cold = [obs('A', 1, split='test', **holo), obs('B', 2, split='train', split_strategy='target')]
        shared = data.cross_audit(target_cold, [obs('Z', 3, split='train', target_group='Z', **holo)])
        self.assertEqual((shared['strategy'], shared['overlap']['structure_id']['count'], shared['leakage']), ('target', 1, True))
        with self.assertRaisesRegex(ValueError, 'no train rows'):
            data.cross_audit(mine, [obs('Z', 0)])


if __name__ == '__main__':
    unittest.main()
