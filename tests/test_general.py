"""General screening workflow: small real inputs, no Protenix dependency."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from fragment_ft import data


def observations():
    return [dict(sample_id=f'{target}_{i}', target_id=target, target_group=target,
                 ligand_id=f'L{i}', smiles='C' * (i+1), chem_group=f'G{i}', label=str(i % 2),
                 split='', structure_id='', protein_chain_id='', ligand_chain_id='',
                 assay_id=f'{target}_screen', assay_type='xray', endpoint='hit',
                 campaign_id='study', source='example', source_url='https://example.org/study',
                 quality='pass', weight='1', concentration='10', concentration_unit='mM')
            for target in ('A', 'B', 'C', 'D', 'E', 'F') for i in range(12)]


class GeneralDataTests(unittest.TestCase):
    def test_lit_import_joins_compounds_and_retains_both_outcomes(self):
        from fragment_ft.sources import import_lit
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'active.smi').write_text('CC 101\n')
            (root/'inactive.smi').write_text('CO 102\n')
            (root/'compounds.csv').write_text('source_id,ligand_id,smiles,chem_group\n101,L1,CC,G1\n102,L2,CO,G2\n')
            constants = {k: v for k, v in observations()[0].items()
                         if k not in ('sample_id', 'ligand_id', 'smiles', 'chem_group', 'label')}
            config = {'constants': constants, 'outcomes': {'Active': '1', 'Inactive': '0'}}
            result = import_lit(root/'active.smi', root/'inactive.smi', config,
                                root/'compounds.csv', root/'out')
            self.assertEqual([(r['ligand_id'], r['label']) for r in result], [('L1', '1'), ('L2', '0')])
            config['outcomes'] = {'Active': 'uncertain', 'Inactive': 'unknown'}
            reviewed = import_lit(root/'active.smi', root/'inactive.smi', config,
                                  root/'compounds.csv', root/'reviewed')
            self.assertEqual([r['label'] for r in reviewed], ['uncertain', 'unknown'])
            self.assertFalse(any(data.trainable(r) for r in reviewed))

    def test_download_checks_integrity_and_never_overwrites(self):
        from fragment_ft.sources import download
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import hashlib
        import threading
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                if self.path == '/truncated':
                    self.send_header('Content-Length', '100')
                self.end_headers(); self.wfile.write(b'public data')
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); url = f'http://127.0.0.1:{server.server_port}/data'
                digest = hashlib.sha256(b'public data').hexdigest()
                self.assertEqual(download(url, root/'data', digest)['sha256'], digest)
                with self.assertRaises(FileExistsError): download(url, root/'data')
                with self.assertRaisesRegex(ValueError, 'SHA256'): download(url, root/'bad', '0'*64)
                self.assertFalse((root/'bad').exists())
                self.assertTrue((root/'bad.partial').exists())
                with self.assertRaisesRegex(ValueError, 'Content-Length'):
                    download(url.replace('/data', '/truncated'), root/'truncated')
                self.assertFalse((root/'truncated').exists())
                self.assertFalse((root/'truncated.receipt.json').exists())
        finally:
            server.shutdown(); worker.join(); server.server_close()

    def test_quality_and_phenotype_are_not_binding_negatives(self):
        row = observations()[0]
        self.assertTrue(data.trainable(row))
        self.assertFalse(data.trainable(dict(row, quality='fail')))
        self.assertFalse(data.trainable(dict(row, assay_type='phenotypic', endpoint='viability')))
        self.assertFalse(data.trainable(dict(row, label='unknown')))
        self.assertNotEqual(data.task_name(row), data.task_name(dict(row, assay_type='biochemical', endpoint='inhibition')))

    def test_cold_splits_and_double_cold_exclusions(self):
        rows = observations()
        for strategy in ('chemistry', 'target', 'both'):
            result = data.assign_splits(rows, seed=4, validation=.25, test=.25, strategy=strategy)
            data.validate_rows(result, require_splits=True)
            report = data.split_audit(result)
            self.assertEqual(report['strategy'], strategy)
            axes = ['chem_group'] if strategy == 'chemistry' else ['target_group'] if strategy == 'target' else ['chem_group', 'target_group']
            for axis in axes:
                sets = [{r[axis] for r in result if r['split'] == s} for s in ('train', 'val', 'test')]
                self.assertFalse(sets[0] & sets[1] or sets[1] & sets[2] or sets[0] & sets[2])
            self.assertEqual(any(r['split'] == 'excluded' for r in result), strategy == 'both')
        broken = data.assign_splits(rows, strategy='target')
        broken[0]['split'] = 'test' if broken[0]['split'] != 'test' else 'train'
        with self.assertRaisesRegex(ValueError, 'leakage'):
            data.validate_rows(broken, require_splits=True)

    def test_metrics_keep_assays_and_endpoints_separate(self):
        rows = [dict(observations()[i], probability=.9 if i % 2 else .1) for i in range(2)]
        rows += [dict(rows[0], sample_id='other', assay_id='enzyme', assay_type='biochemical', endpoint='inhibition')]
        report = data.prediction_report(rows)
        self.assertEqual(len(report), 2, 'Different experimental endpoints must not share metrics')

    def test_import_maps_observed_outcomes_and_preserves_raw_rows(self):
        from fragment_ft.sources import import_csv
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'screen.csv'
            source.write_text('id,smiles,group,outcome,qc\n1,CC,G1,Active,pass\n2,CO,G2,Inactive,pass\n3,CN,G3,Inactive,fail\n4,CCC,G4,Inconclusive,pass\n')
            constants = {k: v for k, v in observations()[0].items()
                         if k not in ('sample_id', 'ligand_id', 'smiles', 'chem_group', 'label', 'quality')}
            config = {'columns': {'ligand_id': 'id', 'smiles': 'smiles', 'chem_group': 'group', 'quality': 'qc'},
                      'constants': constants, 'outcome_column': 'outcome',
                      'outcomes': {'Active': '1', 'Inactive': '0', 'Inconclusive': 'uncertain'}}
            rows = import_csv(source, config, root/'imported')
            self.assertEqual([r['label'] for r in rows], ['1', '0', '0', 'uncertain'])
            self.assertEqual(sum(data.trainable(r) for r in rows), 2)
            self.assertEqual(len((root/'imported/raw.jsonl').read_text().splitlines()), 4)
            self.assertTrue((root/'imported/provenance.json').exists())
            self.assertEqual(data.read_manifest(root/'imported/manifest.csv'), rows)
            other = dict(config, constants=dict(constants, assay_id='other_screen'))
            other_rows = import_csv(source, other, root/'other')
            data.validate_rows(rows + other_rows)
            source.write_text('id,smiles,group,outcome,outcome,qc\n1,CC,G1,Inconclusive,Inactive,pass\n')
            with self.assertRaisesRegex(ValueError, 'header'):
                import_csv(source, config, root/'duplicate')
            source.write_text('id,smiles,group,outcome,qc\n1,CC,G1,Unseen,pass\n')
            with self.assertRaisesRegex(ValueError, 'outcome'):
                import_csv(source, config, root/'bad')


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch not installed')
class MultiTaskTests(unittest.TestCase):
    def test_multitask_loop_excludes_failed_phenotypic_and_test_observations(self):
        import torch
        from fragment_ft.__main__ import parser
        from fragment_ft.training import FineTuner, run_training
        class Backend(torch.nn.Module):
            c_s = c_z = 2
            def __init__(self):
                super().__init__(); self.model = torch.nn.Linear(2, 2)
            def encode(self, features):
                s = self.model(features['x']); return s, s[:, None] + s[None, :]
        rows = []
        for assay_type, endpoint in [('xray', 'hit'), ('biochemical', 'inhibition')]:
            for i, (split, label) in enumerate([('train', '0'), ('train', '1'), ('val', '0'), ('val', '1')]):
                rows.append(dict(observations()[i], sample_id=f'{assay_type}_{i}',
                                 assay_id=assay_type, assay_type=assay_type, endpoint=endpoint,
                                 split=split, label=label))
        # No packets exist for these rows: any accidental training/evaluation inclusion fails.
        rows += [dict(rows[0], sample_id='bad', quality='fail'),
                 dict(rows[0], sample_id='phenotype', assay_id='cell', assay_type='phenotypic', endpoint='viability'),
                 dict(observations()[10], sample_id='held_out', split='test'),
                 # Validation runs on every val row it keeps, so these fail deterministically if included.
                 dict(rows[2], sample_id='bad_val', quality='fail'),
                 dict(rows[3], sample_id='uncertain_val', label='uncertain')]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            features = {'x': torch.eye(3, 2), 'atom_to_token_idx': torch.arange(3),
                        'is_protein': torch.tensor([1, 1, 0]), 'is_ligand': torch.tensor([0, 0, 1])}
            for row in rows[:8]:
                torch.save({'version': 1, 'kind': 'binding', 'row_hash': data.row_hash(row),
                            'input_feature_dict': features}, root/f'{row["sample_id"]}.binding.pt')
            args = parser().parse_args(['train', 'unused', '--features', str(root),
                '--base-checkpoint', 'unused', '--output', str(root/'run'), '--device', 'cpu',
                '--precision', 'fp32', '--steps', '3', '--eval-every', '3', '--accumulate', '2'])
            tasks = ['biochemical:inhibition', 'xray:hit']
            model = FineTuner(Backend(), hidden=4, task_names=tasks)
            run_training(model, rows, root, args, {'task_names': tasks})
            report = json.loads((root/'run/step_000003.json').read_text())
            self.assertEqual({item['task'] for item in report['validation'].values()}, set(tasks))
            self.assertEqual(len(report['predictions']), 4)
            self.assertTrue(all(r['split'] == 'val' for r in report['predictions']))

    def test_joint_to_frozen_transfer_survives_checkpoint_restore(self):
        import torch
        from fragment_ft.training import FineTuner, checkpoint_payload, initialize_delta, load_delta
        class Backend(torch.nn.Module):
            c_s = c_z = 2
            def __init__(self):
                super().__init__(); self.model = torch.nn.Linear(2, 2)
        pretrained = FineTuner(Backend(), mode='joint', trainable_prefixes=['weight'], hidden=4)
        pretrained.backend.model.weight.data.fill_(7.)
        optimizer = torch.optim.AdamW([p for p in pretrained.parameters() if p.requires_grad])
        payload = checkpoint_payload(pretrained, optimizer, 2, {}, .5)
        frozen = FineTuner(Backend(), hidden=4)
        initialize_delta(frozen, payload)
        optimizer = torch.optim.AdamW(frozen.head.parameters())
        transferred = checkpoint_payload(frozen, optimizer, 1, {}, .4)
        restored = FineTuner(Backend(), hidden=4)
        load_delta(restored, transferred)
        self.assertTrue(torch.equal(restored.backend.model.weight, frozen.backend.model.weight),
                        'Previously trained frozen backbone deltas must not disappear')

    def test_only_requested_output_receives_supervision_and_transfer_preserves_tasks(self):
        import torch
        from fragment_ft.training import FineTuner, checkpoint_payload, initialize_delta, training_loss
        class Backend(torch.nn.Module):
            c_s = c_z = 2
            def __init__(self):
                super().__init__(); self.model = torch.nn.Linear(2, 2)
            def encode(self, f):
                s = self.model(f['x']); return s, s[:, None] + s[None, :]
        f = {'x': torch.ones(3, 2), 'atom_to_token_idx': torch.arange(3),
             'is_protein': torch.tensor([1, 1, 0]), 'is_ligand': torch.tensor([0, 0, 1])}
        tasks = ['biochemical:inhibition', 'xray:hit']
        a = FineTuner(Backend(), hidden=4, task_names=tasks)
        logit, structural = a(f, task='xray:hit')
        training_loss(logit, 1, structural, sample_weight=.5).backward()
        self.assertEqual(a.head[-1].weight.grad[0].abs().sum().item(), 0)
        self.assertGreater(a.head[-1].weight.grad[1].abs().sum().item(), 0)
        optimizer = torch.optim.AdamW(a.head.parameters())
        payload = checkpoint_payload(a, optimizer, 3, {'task_names': tasks}, 1.)
        b = FineTuner(Backend(), hidden=4, task_names=['direct_binding:binding', 'xray:hit'])
        initialize_delta(b, payload)
        self.assertTrue(torch.equal(a.head[-1].weight[1], b.head[-1].weight[1]))
        self.assertTrue(torch.equal(a.head[1].weight, b.head[1].weight))
        disjoint = FineTuner(Backend(), hidden=4, task_names=['direct_binding:binding'])
        initial_output = disjoint.head[-1].weight.detach().clone()
        initialize_delta(disjoint, payload)
        self.assertTrue(torch.equal(disjoint.head[-1].weight, initial_output))
        self.assertTrue(torch.equal(disjoint.head[1].weight, a.head[1].weight))
        with self.assertRaisesRegex(ValueError, 'task'):
            b(f, task='biochemical:inhibition')
