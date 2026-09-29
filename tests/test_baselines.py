"""Baseline contracts, using real small models and no native measurements."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from fragment_ft import baselines, data
from fragment_ft.__main__ import main as parent_main


def observations():
    rows = []
    for i in range(10):
        rows.append(dict(sample_id=f's{i}', target_id='T', ligand_id=f'L{i}',
                         smiles='C' * (i + 1), chem_group=f'G{i}', label=str(i % 2),
                         split='train' if i < 8 else 'test', structure_id='',
                         protein_chain_id='', ligand_chain_id='', target_group='T',
                         assay_id='screen', assay_type='xray', endpoint='hit',
                         campaign_id='study', source='fictional', source_url='https://example.org',
                         quality='pass', weight='1'))
    rows[2]['label'] = 'unknown'
    rows[3]['label'] = 'uncertain'
    rows[4]['quality'] = 'fail'
    rows[5]['weight'] = '0'
    rows[6].update(assay_type='phenotypic', assay_id='phenotype')
    rows[7]['split'] = 'val'
    return rows


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = observations()
        self.manifest = self.root / 'manifest.csv'
        data.write_manifest(self.manifest, self.rows)
        self.parser = argparse.ArgumentParser()
        baselines.register_parser(self.parser.add_subparsers(dest='command', required=True))

    def execute(self, method='prior', extra=(), manifest=None, name='report.json'):
        output = self.root / name
        args = self.parser.parse_args(['baseline', str(manifest or self.manifest), '--method', method,
                                      '--output', str(output), *extra])
        self.assertEqual(baselines.run(args), 0)
        return data.read_json(output)

    def scores(self, metric='iptm', values=(0.2, 0.8)):
        return {'metric': metric, 'provenance': {'source': 'fictional fixture', 'model': 'fixture',
                'score_definition': 'test scalar', 'selection_policy': 'one fixed sample'},
                'scores': [dict(sample_id=row['sample_id'], target_id=row['target_id'],
                                ligand_id=row['ligand_id'], score=value)
                           for row, value in zip(self.rows[8:], values)]}

    def imported(self, payload, metric='iptm', orientation='higher', scale='1', name='imported.json'):
        path = self.root / (name + '.scores')
        data.write_json(path, payload)
        return self.execute('protenix', ['--scores', str(path), '--metric', metric,
                                       '--orientation', orientation, '--score-scale', scale], name=name)

    def test_prior_uses_only_reliable_train_rows(self):
        report = self.execute()
        self.assertEqual(report['training']['fit_sample_ids'], ['s0', 's1'])
        self.assertEqual([p['probability'] for p in report['predictions']], [0.5, 0.5])

    def test_prior_target_rates_and_cold_target_fallback(self):
        self.rows[1].update(target_id='B', target_group='B')
        self.rows[9].update(target_id='B', target_group='B')
        cold = dict(self.rows[8], sample_id='cold', target_id='C', target_group='C')
        changed = self.root / 'targets.csv'
        data.write_manifest(changed, self.rows + [cold])
        report = self.execute(manifest=changed)
        self.assertEqual([p['probability'] for p in report['predictions']], [0.0, 1.0, 0.5])
        self.assertEqual(report['details']['fallback_sample_ids'], ['cold'])
        self.assertEqual(report['details']['target_hit_rates'], {'xray:hit': {'B': 1.0, 'T': 0.0}})
        self.assertEqual(report['details']['task_hit_rates'], {'xray:hit': 0.5})

    def test_prior_does_not_fit_held_out_labels(self):
        first = self.execute()
        for row in self.rows[7:]:
            row['label'] = '0'
        changed = self.root / 'changed.csv'
        data.write_manifest(changed, self.rows)
        second = self.execute(manifest=changed, name='second.json')
        self.assertEqual(first['details'], second['details'])
        self.assertEqual(first['training']['fit_row_hashes'], second['training']['fit_row_hashes'])

    def test_unreliable_selected_rows_are_predicted_but_not_scored(self):
        self.rows[8]['label'] = 'unknown'
        changed = self.root / 'unknown.csv'
        data.write_manifest(changed, self.rows)
        report = self.execute(manifest=changed)
        self.assertEqual(len(report['predictions']), 2)
        self.assertEqual(sum(item['n'] for item in report['metrics_by_target'].values()), 1)

    def test_missing_train_task_fails_instead_of_using_test_labels(self):
        self.rows[0]['quality'] = self.rows[1]['quality'] = 'fail'
        changed = self.root / 'no-train.csv'
        data.write_manifest(changed, self.rows)
        with self.assertRaisesRegex(ValueError, 'No reliable train'):
            self.execute(manifest=changed)

    def test_declared_split_leakage_is_rejected(self):
        self.rows[8].update(smiles='C', chem_group='G0')
        changed = self.root / 'leak.csv'
        data.write_manifest(changed, self.rows)
        with self.assertRaisesRegex(ValueError, 'split leakage'):
            self.execute(manifest=changed)

    @unittest.skipUnless(importlib.util.find_spec('rdkit'), 'RDKit unavailable')
    def test_ligand_fit_is_deterministic_and_excludes_held_out_statistics(self):
        first = self.execute('ligand')
        self.rows[7].update(smiles='N', label='0')
        self.rows[8].update(smiles='O', label='1')
        # Invalid SMILES in unreliable train rows must not reach descriptor fitting.
        self.rows[2]['smiles'] = 'invalid'
        changed = self.root / 'changed.csv'
        data.write_manifest(changed, list(reversed(self.rows)))
        second = self.execute('ligand', manifest=changed, name='second.json')
        self.assertEqual(first['details'], second['details'])
        self.assertEqual(first['training']['fit_sample_ids'], ['s0', 's1'])
        scores = {p['sample_id']: p['probability'] for p in second['predictions']}
        self.assertEqual(first['predictions'][1]['probability'], scores['s9'])
        self.assertNotEqual(first['predictions'][0]['probability'], scores['s8'])

    @unittest.skipUnless(importlib.util.find_spec('rdkit'), 'RDKit unavailable')
    def test_ligand_single_class_requires_prior(self):
        self.rows[1]['label'] = '0'
        changed = self.root / 'single.csv'
        data.write_manifest(changed, self.rows)
        with self.assertRaisesRegex(ValueError, 'both train classes'):
            self.execute('ligand', manifest=changed)

    def test_missing_rdkit_is_actionable_without_affecting_prior(self):
        with mock.patch.dict(sys.modules, {'rdkit': None}):
            with self.assertRaisesRegex(ImportError, 'existing RDKit'):
                self.execute('ligand')
            self.execute('prior')

    def test_confidence_direction_scaling_and_raw_provenance(self):
        for metric, values, direction, scale, expected in (
                ('iptm', (0.2, 0.8), 'higher', '1', [0.2, 0.8]),
                ('plddt', (20, 80), 'higher', '100', [0.2, 0.8]),
                ('plddt', (0.2, 0.8), 'higher', '1', [0.2, 0.8]),
                ('pae', (0, 30), 'lower', '10', [1.0, 0.25])):
            with self.subTest(metric=metric, scale=scale):
                payload = self.scores(metric, values)
                payload['scores'].reverse()
                report = self.imported(payload, metric, direction, scale, name=f'{metric}-{scale}.json')
                self.assertEqual([p['probability'] for p in report['predictions']], expected)
                self.assertEqual(report['raw_scores'], payload['scores'])
                self.assertEqual(report['provenance'], payload['provenance'])
                self.assertEqual(report['training']['fit_sample_ids'], [])
                for prediction in report['predictions']:
                    self.assertNotIn('score', prediction)
                    self.assertNotIn('provenance', prediction)

    def test_import_rejects_matching_failures(self):
        for case in ('duplicate', 'missing', 'extra', 'target', 'ligand', 'case'):
            with self.subTest(case=case):
                payload = self.scores()
                if case == 'duplicate':
                    payload['scores'].append(payload['scores'][0].copy())
                elif case == 'missing':
                    payload['scores'].pop()
                elif case == 'extra':
                    payload['scores'].append(dict(payload['scores'][0], sample_id='s0'))
                elif case == 'case':
                    payload['scores'][0]['sample_id'] = 'S8'
                else:
                    payload['scores'][0][case + '_id'] = 'wrong'
                with self.assertRaises(ValueError):
                    self.imported(payload, name=f'{case}.json')

    def test_import_rejects_invalid_numbers_and_ranges(self):
        for index, value in enumerate((-1, 1.1, True, '0.5', None, 10**400)):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.imported(self.scores(values=(value, 0.8)), name=f'invalid-{index}.json')
        for index, value in enumerate((float('nan'), float('inf'), -float('inf'))):
            payload = self.scores(values=(value, 0.8))
            path = self.root / f'nonfinite-{index}.json'
            with path.open('w') as stream:
                json.dump(payload, stream)
            with self.assertRaises(ValueError):
                self.execute('protenix', ['--scores', str(path), '--metric', 'iptm',
                                         '--orientation', 'higher', '--score-scale', '1'])

    def test_import_requires_explicit_valid_direction_scale_and_schema(self):
        for index, (metric, direction, scale) in enumerate((('iptm', 'lower', '1'),
                ('iptm', 'higher', '100'), ('pae', 'higher', '10'), ('pae', 'lower', '0'),
                ('pae', 'lower', 'nan'), ('plddt', 'higher', '2'))):
            with self.subTest(metric=metric, direction=direction, scale=scale):
                with self.assertRaises(ValueError):
                    self.imported(self.scores(metric), metric, direction, scale, name=f'flags-{index}.json')
        with self.assertRaisesRegex(ValueError, 'requires --scores'):
            self.execute('protenix')
        for case in ('metric', 'provenance', 'extra'):
            payload = self.scores()
            if case == 'metric':
                payload['metric'] = 'pae'
            elif case == 'provenance':
                payload['provenance'].pop('selection_policy')
            else:
                payload['scores'][0]['probability'] = 0.2
            with self.assertRaises(ValueError):
                self.imported(payload, name=f'schema-{case}.json')

    def test_reports_compare_through_parent_cli(self):
        self.execute()
        self.imported(self.scores())
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = parent_main(['compare', str(self.root / 'report.json'),
                                str(self.root / 'imported.json'), '--output', str(self.root / 'compare.json')])
        self.assertEqual(code, 0)
        self.assertEqual(len(data.read_json(self.root / 'compare.json')), 2)

    def test_main_cli_and_overwrite_error(self):
        command = [sys.executable, '-m', 'fragment_ft', 'baseline', str(self.manifest),
                   '--method', 'prior', '--output', str(self.root / 'cli.json')]
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(data.read_json(self.root / 'cli.json')['baseline'], 'prior')
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 2)
        self.assertIn('Refusing to overwrite', result.stderr)


if __name__ == '__main__':
    unittest.main()
