"""Run with python -m unittest discover -s tests -v; no Protenix required."""
import copy
from contextlib import redirect_stderr
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

from fragment_ft.data import read_manifest, assign_splits, validate_rows, metrics, make_inputs


def rows():
    return [dict(sample_id=f'{t}_{i}', target_id=t, ligand_id=f'F{i}',
                 smiles=f'C{"C" * i}', chem_group=f'G{i}',
                 label=str(i % 2), split='', structure_id='',
                 protein_chain_id='', ligand_chain_id='')
            for t in ('A', 'B') for i in range(8)]


class ManifestTests(unittest.TestCase):
    def test_shared_chemistry_stays_together_and_is_order_independent(self):
        a = assign_splits(rows(), seed=42, validation=0.25, test=0.25)
        b = assign_splits(list(reversed(rows())), seed=42, validation=0.25, test=0.25)
        self.assertEqual({r['sample_id']: r['split'] for r in a},
                         {r['sample_id']: r['split'] for r in b})
        validate_rows(a, require_splits=True)
        self.assertEqual({r['split'] for r in a}, {'train', 'val', 'test'})
        a[0]['split'] = 'test' if a[0]['split'] != 'test' else 'train'
        with self.assertRaisesRegex(ValueError, 'split'):
            validate_rows(a, require_splits=True)

    def test_rejects_inconsistent_identity_negative_structure_and_unknown_label(self):
        for field, value in [('smiles', 'O'), ('chem_group', 'different')]:
            a = rows(); a[8][field] = value
            with self.assertRaises(ValueError): validate_rows(a)
        a = rows(); a[0]['structure_id'] = 'fake'
        with self.assertRaisesRegex(ValueError, 'positive'): validate_rows(a)
        a = rows(); a[0]['label'] = 'failed'
        with self.assertRaisesRegex(ValueError, 'label'): validate_rows(a)

    def test_uncertain_and_no_files_required_for_manifest(self):
        a = rows(); a[0]['label'] = 'uncertain'
        validate_rows(a)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'manifest.csv'
            with p.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=a[0]); writer.writeheader(); writer.writerows(a)
            self.assertEqual(read_manifest(p), a)

    def test_input_context_shared_and_no_labels_or_holo_paths(self):
        from fragment_ft.data import binding_key
        a = rows(); a[1].update(structure_id='holo', protein_chain_id='A', ligand_chain_id='B')
        inputs = make_inputs(a, {'A': {'sequence': 'ACD', 'count': 1},
                                 'B': {'sequence': 'DEF', 'count': 1}})
        self.assertNotIn('holo', json.dumps(inputs))
        self.assertNotIn('label', json.dumps(inputs))
        self.assertEqual(inputs[0]['sequences'][0], inputs[1]['sequences'][0])
        self.assertEqual(binding_key(a[0]), binding_key(dict(a[0], label='1', split='test')))

    def test_compare_requires_identical_evaluation_examples(self):
        from fragment_ft.__main__ import main
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [dict(row, split='test', probability=0.8 if row['label'] == '1' else 0.2)
                       for row in rows()]
            paths = [root/'baseline.json', root/'synthetic.json']
            for path in paths:
                path.write_text(json.dumps({'predictions': records}))
            self.assertEqual(main(['compare', *map(str, paths), '--output', str(root/'comparison.json')]), 0)
            comparison = json.loads((root/'comparison.json').read_text())
            self.assertEqual(comparison[str(paths[0])]['metrics_by_target']['A']['average_precision'], 1.)
            records[0]['label'] = '1'
            paths[1].write_text(json.dumps({'predictions': records}))
            with redirect_stderr(io.StringIO()):
                self.assertEqual(main(['compare', *map(str, paths), '--output', str(root/'invalid.json')]), 2)
            self.assertFalse((root/'invalid.json').exists())

    def test_cache_provenance_rejects_changed_content(self):
        from fragment_ft.data import file_hash, verify_features, write_json
        with tempfile.TemporaryDirectory() as tmp:
            a = rows()[:1]; a[0]['split'] = 'train'
            packet = Path(tmp) / f'{a[0]["sample_id"]}.binding.pt'
            packet.write_bytes(b'original')
            write_json(Path(tmp) / 'preparation.json', {'artifacts': {packet.name: file_hash(packet)}})
            verify_features(tmp, a)
            packet.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed'): verify_features(tmp, a)

    def test_metrics_ties_and_missing_class(self):
        result = metrics([1, 0, 1, 0], [0.5] * 4, k=2)
        self.assertAlmostEqual(result['average_precision'], 0.5)
        self.assertAlmostEqual(result['brier'], 0.25)
        self.assertIsNone(metrics([0, 0], [0.1, 0.2])['average_precision'])
        with self.assertRaises(ValueError): metrics([1], [float('nan')])


HAS_TORCH = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(HAS_TORCH, 'PyTorch is not installed in this Python environment')
class TrainingTests(unittest.TestCase):
    def test_frozen_stochastic_encoder_is_reproducible_after_restore(self):
        import torch
        from fragment_ft.training import FineTuner, checkpoint_payload, load_delta
        class Stochastic(torch.nn.Module):
            c_s, c_z = 2, 2
            def __init__(self):
                super().__init__(); self.model = torch.nn.Linear(2, 2)
            def encode(self, features):
                s = self.model(features['x']) + torch.rand(3, 2)
                return s, s[:, None, :] + s[None, :, :]
        f = {'x': torch.ones(3, 2), 'atom_to_token_idx': torch.arange(3),
             'is_protein': torch.tensor([1, 1, 0]), 'is_ligand': torch.tensor([0, 0, 1])}
        torch.manual_seed(4)
        a = FineTuner(Stochastic(), hidden=4)
        b = FineTuner(Stochastic(), hidden=4)
        b.backend.load_state_dict(a.backend.state_dict())
        optimizer = torch.optim.AdamW(a.head.parameters())
        load_delta(b, checkpoint_payload(a, optimizer, 0, {}, 1.0))
        a.eval(); b.eval()
        before = torch.get_rng_state().clone()
        first = a(f, cache_key='same')[0]
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        torch.rand(10)
        second = b(f, cache_key='same')[0]
        self.assertTrue(torch.equal(first, second))

    def test_apo_mapping_is_injective_and_integral(self):
        from fragment_ft.synthetic import residue_mapping
        self.assertEqual(residue_mapping([1, 2], {'residue_offset': 100}), {1: 101, 2: 102})
        for settings in ({'residue_map': {'2': 1}}, {'residue_offset': 1.5},
                         {'residue_map': {'1': 1.2}}, {'residue_map': {'1': True}}):
            with self.assertRaises(ValueError): residue_mapping([1, 2], settings)

    def test_actual_loop_synthetic_comparison_resume_and_negative_guard(self):
        import torch
        from fragment_ft.__main__ import parser
        from fragment_ft.data import row_hash
        from fragment_ft.training import FineTuner, run_training, load_packet

        class Backend(torch.nn.Module):
            c_s, c_z = 2, 2
            def __init__(self):
                super().__init__(); self.model = torch.nn.Linear(2, 2, bias=False)
                self.kinds = []
            def encode(self, features):
                s = self.model(features['x'])
                return s, s[:, None, :] + s[None, :, :]
            def structure_loss(self, batch, step):
                self.kinds.append(batch['kind'])
                return self.model(batch['x']).square().mean()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cache = root / 'cache'; cache.mkdir()
            a = rows()
            for row in a:
                i = int(row['ligand_id'][1:])
                row['split'] = 'train' if i < 4 else 'val' if i < 6 else 'test'
                if row['label'] == '1':
                    row.update(structure_id=row['sample_id'], protein_chain_id='A', ligand_chain_id='B')
                features = {'x': torch.tensor([[1., 0.], [0., 1.], [0.5, 0.5]]),
                            'atom_to_token_idx': torch.tensor([0, 1, 2]),
                            'is_protein': torch.tensor([1, 1, 0]),
                            'is_ligand': torch.tensor([0, 0, 1])}
                coordinates = torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
                kinds = ['binding', 'structure' if row['label'] == '1' else 'synthetic']
                for kind in kinds:
                    packet = {'version': 1, 'kind': kind, 'row_hash': row_hash(row),
                              'input_feature_dict': features, 'x': features['x'],
                              'label_dict': {'coordinate': coordinates, 'coordinate_mask': torch.ones(3)},
                              'label_full_dict': {'coordinate': coordinates.clone()}}
                    torch.save(packet, cache / f'{row["sample_id"]}.{kind}.pt')
            with self.assertRaisesRegex(ValueError, 'positive'):
                load_packet(cache, a[0], 'structure', torch.device('cpu'))
            args = parser().parse_args(['train', 'unused.csv', '--features', str(cache),
                                        '--base-checkpoint', 'unused.pt', '--output', str(root/'full'),
                                        '--mode', 'joint', '--trainable-prefix', 'weight',
                                        '--negative-mode', 'synthetic', '--device', 'cpu',
                                        '--precision', 'fp32', '--steps', '3', '--eval-every', '1',
                                        '--accumulate', '2'])
            torch.manual_seed(12)
            model = FineTuner(Backend(), 'joint', ['weight'], hidden=4)
            initial = copy.deepcopy(model.state_dict())
            run_training(model, a, cache, args, metadata={'test': 'resume'})
            self.assertIn('structure', model.backend.kinds)
            self.assertIn('synthetic', model.backend.kinds)
            with (root/'full/step_000003.json').open() as stream:
                report = json.load(stream)
            self.assertEqual(set(report['validation']), {'A', 'B'})
            self.assertTrue(all(r['split'] == 'val' for r in report['predictions']))
            resumed = FineTuner(Backend(), 'joint', ['weight'], hidden=4)
            resumed.load_state_dict(initial)
            args.resume = str(root/'full/step_000001.pt'); args.output = str(root/'resumed')
            run_training(resumed, a, cache, args, metadata={'test': 'resume'})
            for key, tensor in model.state_dict().items():
                self.assertTrue(torch.equal(tensor, resumed.state_dict()[key]), key)

    def test_synthetic_placement_preserves_geometry_and_changes_only_ligand(self):
        import torch
        from fragment_ft.synthetic import detached_coordinates
        coordinates = torch.tensor([[0., 0., 0.], [3., 0., 0.], [0., 2., 0.],
                                    [1., 1., 1.], [2., 1., 1.]])
        ligand = torch.tensor([False, False, False, True, True])
        protein = ~ligand
        a = detached_coordinates(coordinates, protein, ligand, clearance=40., seed=4)
        b = detached_coordinates(coordinates, protein, ligand, clearance=40., seed=5)
        self.assertTrue(torch.equal(a[protein], coordinates[protein]))
        self.assertTrue(torch.allclose(torch.cdist(a[ligand], a[ligand]),
                                       torch.cdist(coordinates[ligand], coordinates[ligand]), atol=1e-5))
        self.assertGreaterEqual(torch.cdist(a[protein], a[ligand]).min().item(), 40. - 1e-5)
        self.assertTrue(torch.equal(a, detached_coordinates(coordinates, protein, ligand, 40., 4)))
        self.assertFalse(torch.equal(a, b))
        with self.assertRaises(ValueError): detached_coordinates(coordinates, protein, ligand, -1., 4)

    def test_negative_binding_and_positive_structure_gradients(self):
        import torch
        from fragment_ft.training import FineTuner, training_loss, load_delta, checkpoint_payload

        class Backend(torch.nn.Module):
            c_s, c_z = 2, 2
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Linear(2, 2, bias=False)
                self.structures = []
            def encode(self, features):
                s = self.model(features['x'])
                z = s[:, None, :] + s[None, :, :]
                return s, z
            def structure_loss(self, batch, step):
                self.structures.append(batch['sample_id'])
                return self.model(batch['x']).square().mean()

        features = {'x': torch.tensor([[1., 0.], [0., 1.], [0.5, 0.5]]),
                    'atom_to_token_idx': torch.tensor([0, 0, 1, 2]),
                    'is_protein': torch.tensor([1, 1, 1, 0]),
                    'is_ligand': torch.tensor([0, 0, 0, 1])}
        torch.manual_seed(8)
        model = FineTuner(Backend(), mode='joint', trainable_prefixes=['weight'], hidden=4)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
        positive = {'sample_id': 'positive', 'x': torch.ones(2, 2)}
        logit, structural = model(features, positive, 0)
        loss = training_loss(logit, 0, structural, structure_weight=1.0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model.backend.model.weight.grad.abs().sum().item(), 0)
        self.assertEqual(model.backend.structures, ['positive'])
        optimizer.step()
        payload = checkpoint_payload(model, optimizer, step=1, metadata={'example': True}, best=0.5)
        restored = FineTuner(Backend(), mode='joint', trainable_prefixes=['weight'], hidden=4)
        load_delta(restored, payload)
        model.eval(); restored.eval()
        self.assertTrue(torch.equal(model(features, None, 0)[0], restored(features, None, 0)[0]))
        # Frozen mode updates only the small head and never calls structural training.
        frozen = FineTuner(Backend(), mode='head', trainable_prefixes=[], hidden=4)
        frozen.train()
        output, structural = frozen(features, None, 0)
        training_loss(output, 1, structural, structure_weight=0).backward()
        self.assertIsNone(frozen.backend.model.weight.grad)
        self.assertFalse(frozen.backend.training)
        self.assertEqual(frozen.backend.structures, [])
        with self.assertRaisesRegex(ValueError, 'head'):
            frozen(features, positive, 0)

    def test_token_pooling_counts_tokens_not_atoms(self):
        import torch
        from fragment_ft.training import pool_interface
        s = torch.tensor([[1., 2.], [3., 4.], [8., 10.]])
        z = torch.zeros(3, 3, 2); z[0, 2] = 2; z[1, 2] = 4
        f = {'atom_to_token_idx': torch.tensor([0, 0, 1, 2]),
             'is_protein': torch.tensor([1, 1, 1, 0]),
             'is_ligand': torch.tensor([0, 0, 0, 1])}
        pooled = pool_interface(s, z, f)
        self.assertTrue(torch.equal(pooled[:4], torch.tensor([2., 3., 8., 10.])))
        bad = copy.deepcopy(f); bad['is_ligand'][:] = 0
        with self.assertRaisesRegex(ValueError, 'ligand'): pool_interface(s, z, bad)


if __name__ == '__main__':
    unittest.main()
