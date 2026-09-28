"""Training-loop policies with a tiny CPU backend: loss routing, packets, seeds, selection, resume, sharding."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest import mock

from fragment_ft import data

HAS_TORCH = importlib.util.find_spec('torch') is not None


def backend():
    import torch
    class Backend(torch.nn.Module):
        c_s = c_z = 2
        def __init__(self):
            super().__init__(); self.model = torch.nn.Linear(2, 2, bias=False)
        def encode(self, features):
            s = self.model(features['x']); return s, s[:, None] + s[None, :]
        def structure_loss(self, batch, step):
            return self.model(batch['x']).square().mean()
    return Backend()


def features(i=0):
    import torch
    return {'x': torch.tensor([[1., 0.], [0., 1.], [i / 10, 1 - i / 10]]), 'atom_to_token_idx': torch.arange(3),
            'is_protein': torch.tensor([1, 1, 0]), 'is_ligand': torch.tensor([0, 0, 1])}


def obs(i, split, label, **changes):
    return dict(sample_id=f'A_{i}', target_id='A', target_group='A', ligand_id=f'L{i}', smiles='C' * (i + 1),
                chem_group=f'G{i}', label=label, split=split, structure_id='', protein_chain_id='',
                ligand_chain_id='', assay_id='A_screen', assay_type='xray', endpoint='hit', campaign_id='study',
                source='example', source_url='https://example.org/study', quality='pass') | changes


def corpus(root):
    """Four train and four val rows with binding_key packets; the untrainable rows have no packets."""
    import torch
    rows = [obs(i, 'train' if i < 4 else 'val', str(i % 2)) for i in range(8)]
    for i, row in enumerate(rows):
        torch.save({'version': 1, 'kind': 'binding', 'binding_key': data.binding_key(row),
                    'input_feature_dict': features(i)}, Path(root) / data.packet_name(row, 'binding'))
    return rows + [obs(8, 'train', '1', quality='fail'), obs(9, 'train', 'uncertain'),
                   obs(10, 'test', '0'), obs(11, 'val', '1', quality='fail')]


def train_args(root, output, steps, *extra):
    """Later flags in extra override the defaults here (argparse keeps the last value)."""
    from fragment_ft.__main__ import parser
    return parser().parse_args(['train', 'unused.csv', '--features', str(root), '--base-checkpoint', 'unused.pt',
                                '--output', str(output), '--device', 'cpu', '--precision', 'fp32',
                                '--steps', str(steps), '--eval-every', '2', '--accumulate', '2', '--head-lr', '0.05',
                                *map(str, extra)])


def distributed_worker(rank, root, port):
    import torch
    from fragment_ft.training import FineTuner, run_training, training_loss
    losses = []
    def record(*args, **kwargs):
        loss = training_loss(*args, **kwargs); losses.append(loss.item()); return loss
    os.environ.update(WORLD_SIZE='2', RANK=str(rank), LOCAL_RANK=str(rank), MASTER_ADDR='127.0.0.1',
                      MASTER_PORT=str(port))
    torch.manual_seed(0)
    model = FineTuner(backend(), hidden=4)
    args = train_args(root, Path(root) / 'run', 2, '--dist-timeout-minutes', 1)
    with redirect_stdout(io.StringIO()), mock.patch('fragment_ft.training.training_loss', side_effect=record):
        run_training(model, data.read_manifest(Path(root) / 'rows.csv'), root, args, {'test': 'distributed'})
    (Path(root) / f'losses_{rank}.json').write_text(json.dumps(losses))


@unittest.skipUnless(HAS_TORCH, 'PyTorch is not installed in this Python environment')
class TrainingTests(unittest.TestCase):
    def test_training_loss_is_the_weighted_sum_of_both_terms(self):
        import torch
        from fragment_ft.training import training_loss
        logit, structural = torch.tensor(.3), torch.tensor(.7)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, torch.tensor(1.)).item()
        for weight, sample_weight in ((0., 1.), (1., 1.), (2., .5)):
            self.assertAlmostEqual(training_loss(logit, 1, structural, weight, sample_weight).item(),
                                   sample_weight * bce + weight * .7, places=6)

    def test_structural_term_reaches_backbone_and_synthetic_weight_scales(self):
        import torch
        from fragment_ft.training import FineTuner, training_loss
        positive = {'x': torch.ones(2, 2)}
        def run(structure_weight):
            torch.manual_seed(0)
            model = FineTuner(backend(), 'joint', ['weight'], hidden=4)
            logit, structural = model(features(), positive, 0)
            training_loss(logit, 0, structural, structure_weight).backward()
            return model, model.backend.model.weight.grad.clone()
        _, binding_only = run(0.)
        model, both = run(1.)
        model.zero_grad()
        model.backend.structure_loss(positive, 0).backward()
        self.assertGreater(model.backend.model.weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.allclose(both - binding_only, model.backend.model.weight.grad, atol=1e-6))
        synthetic = {'x': torch.tensor([[2., 0.], [0., 1.]])}
        one = model(features(), None, 0, synthetic_structure=synthetic, synthetic_weight=1.)[1]
        self.assertAlmostEqual(model(features(), None, 0, synthetic_structure=synthetic, synthetic_weight=2.5)[1].item(),
                               2.5 * one.item(), places=5)

    def test_head_logits_stay_fp32_under_bf16_autocast(self):
        import torch
        from fragment_ft.training import FineTuner
        model = FineTuner(backend(), hidden=4)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            logit, _ = model(features())
        self.assertEqual(logit.dtype, torch.float32)

    def test_binding_packets_match_binding_key_or_legacy_row_hash(self):
        import torch
        from fragment_ft.training import load_packet
        cpu = torch.device('cpu')
        row = obs(0, 'train', '1', structure_id='S', protein_chain_id='A', ligand_chain_id='B')
        relabeled = dict(row, label='0', split='val', assay_id='other')
        with tempfile.TemporaryDirectory() as tmp:
            def save(kind, **header):
                torch.save(dict(version=1, kind=kind, input_feature_dict=features(), **header),
                           Path(tmp) / data.packet_name(row, kind))
            save('binding', binding_key=data.binding_key(row))
            self.assertIn('x', load_packet(tmp, relabeled, 'binding', cpu)['input_feature_dict'])
            with self.assertRaisesRegex(ValueError, 'Stale'):
                load_packet(tmp, dict(row, smiles='CCO'), 'binding', cpu)
            save('binding', row_hash=data.row_hash(row))
            load_packet(tmp, row, 'binding', cpu)
            with self.assertRaisesRegex(ValueError, 'Stale'):
                load_packet(tmp, relabeled, 'binding', cpu)
            save('structure', binding_key=data.binding_key(row))
            with self.assertRaisesRegex(ValueError, 'Stale'):
                load_packet(tmp, row, 'structure', cpu)

    def test_head_mode_skips_packet_load_on_pooled_cache_hit(self):
        import torch
        from fragment_ft.training import FineTuner, predict
        cpu = torch.device('cpu')
        with tempfile.TemporaryDirectory() as tmp:
            row = corpus(tmp)[4]
            model = FineTuner(backend(), hidden=4)
            first = predict(model, [row], tmp, cpu)
            (Path(tmp) / data.packet_name(row, 'binding')).unlink()
            self.assertEqual(predict(model, [row], tmp, cpu), first)
            with self.assertRaises(FileNotFoundError):
                predict(FineTuner(backend(), 'joint', ['weight'], hidden=4), [row], tmp, cpu)

    def test_micro_seeds_of_nearby_base_seeds_do_not_overlap(self):
        from fragment_ft.training import micro_seed
        streams = [{micro_seed(seed, step, micro, rank) for step in range(50) for micro in range(4) for rank in range(4)}
                   for seed in (42, 43)]
        self.assertEqual(len(streams[0]), 800)
        self.assertFalse(streams[0] & streams[1])
        self.assertNotIn(micro_seed('placement', 42, 10000, 0, 0, 0), streams[0])

    def test_save_checkpoint_refuses_existing_file_or_partial(self):
        from fragment_ft.training import save_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_checkpoint(root / 'a.pt', {'step': 1})
            with self.assertRaises(FileExistsError):
                save_checkpoint(root / 'a.pt', {'step': 2})
            (root / 'b.partial').touch()
            with self.assertRaises(FileExistsError):
                save_checkpoint(root / 'b.pt', {'step': 1})
            self.assertFalse((root / 'b.pt').exists())

    def test_head_run_selection_window_resume_and_refusals(self):
        import torch
        from fragment_ft.training import FineTuner, micro_seed, run_training, seed_step, training_loss
        def run(output, steps, *extra):
            torch.manual_seed(0)
            model = FineTuner(backend(), hidden=4)
            run_training(model, rows, root, train_args(root, output, steps, '--select-metric', 'log_loss', *extra),
                         {'test': 'resume'})
            return model
        def reports(output):
            return [json.loads(path.read_text()) for path in sorted(Path(output).glob('step_*.json'))]
        losses = []
        def record(*args, **kwargs):
            loss = training_loss(*args, **kwargs); losses.append(loss.item()); return loss
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp); rows = corpus(root)
            with mock.patch('fragment_ft.training.sample_row', wraps=data.sample_row) as sampler, \
                    mock.patch('fragment_ft.training.seed_step', wraps=seed_step) as seeder, \
                    mock.patch('fragment_ft.training.training_loss', side_effect=record):
                full = run(root / 'full', 6)
            self.assertEqual([c[0][0] for c in seeder.call_args_list[:3]],
                             [micro_seed(42, 0, 0, 0), micro_seed(42, 0, 1, 0), micro_seed(42, 1, 0, 0)])
            # Untrainable train rows never reach the sampler; untrainable val rows have no packet to load.
            self.assertEqual({r['sample_id'] for r in sampler.call_args[0][1]}, {'A_0', 'A_1', 'A_2', 'A_3'})
            evals = reports(root / 'full')
            self.assertEqual([r['step'] for r in evals], [2, 4, 6])
            for i, report in enumerate(evals):
                self.assertAlmostEqual(report['mean_train_loss_since_last_eval'], sum(losses[4*i:4*i+4]) / 4)
                self.assertEqual([p['sample_id'] for p in report['predictions']], ['A_4', 'A_5', 'A_6', 'A_7'])
                previous = [r['macro_validation_log_loss'] for r in evals[:i]]
                self.assertEqual(report['best_so_far'], report['macro_validation_log_loss'] < min(previous, default=math.inf))
                self.assertIsNotNone(report['macro_validation_average_precision'])
            payload = torch.load(root / 'full/step_000006.pt', weights_only=True)
            best = min(r['macro_validation_log_loss'] for r in evals)
            self.assertEqual((payload['select_metric'], payload['best_selection_score'], payload['best_validation_loss']),
                             ('log_loss', best, best))
            self.assertEqual(payload['metadata']['seed_scheme'], 2)
            start = root / 'full/step_000002.pt'
            with self.assertRaisesRegex(FileExistsError, 'new --output'):
                run(root / 'full', 6, '--resume', start)
            self.assertFalse((root / 'full/run_from_000002.json').exists())
            (root / 'crashed').mkdir(); (root / 'crashed/step_000004.partial').touch()
            with self.assertRaisesRegex(FileExistsError, 'step_000004.partial.*new --output'):
                run(root / 'crashed', 6, '--resume', start)
            legacy = torch.load(start, weights_only=True)
            del legacy['metadata']['seed_scheme']
            torch.save(legacy, root / 'legacy.pt')
            with self.assertRaisesRegex(ValueError, '--init-checkpoint'):
                run(root / 'legacy', 6, '--resume', root / 'legacy.pt')
            resumed = run(root / 'resumed', 6, '--resume', start)
            for a, b in zip(evals[1:], reports(root / 'resumed')):
                self.assertEqual((a['best_so_far'], a['macro_validation_log_loss']),
                                 (b['best_so_far'], b['macro_validation_log_loss']))
            for key, tensor in full.state_dict().items():
                self.assertTrue(torch.equal(tensor, resumed.state_dict()[key]), key)
            # Default metric (higher AP wins), bf16 falling back to fp32 on CPU, crash recovery into the same --output.
            ap = ('--select-metric', 'average_precision', '--precision', 'bf16')
            run(root / 'ap', 4, *ap)
            run(root / 'ap', 6, *ap, '--resume', root / 'ap/step_000004.pt')
            evals = reports(root / 'ap')
            self.assertEqual([r['step'] for r in evals], [2, 4, 6])
            for i, report in enumerate(evals):
                previous = [r['macro_validation_average_precision'] for r in evals[:i]]
                self.assertEqual(report['best_so_far'],
                                 report['macro_validation_average_precision'] > max(previous, default=-math.inf))
            payload = torch.load(root / 'ap/step_000006.pt', weights_only=True)
            self.assertEqual((payload['select_metric'], payload['best_selection_score'], payload['best_validation_loss']),
                             ('average_precision', max(r['macro_validation_average_precision'] for r in evals),
                              min(r['macro_validation_log_loss'] for r in evals)))
            self.assertEqual(payload['metadata']['effective_precision'], 'fp32')

    def test_synthetic_placement_uses_hashed_seed(self):
        import torch
        from fragment_ft.training import FineTuner, micro_seed, run_training
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp); rows = corpus(root)
            for row in rows[:4]:
                kind = 'structure' if row['label'] == '1' else 'synthetic'
                if kind == 'structure':  # binding_key ignores holo metadata, so binding packets stay valid
                    row.update(structure_id=row['sample_id'], protein_chain_id='A', ligand_chain_id='B')
                torch.save({'version': 1, 'kind': kind, 'row_hash': data.row_hash(row), 'x': features()['x']},
                           root / data.packet_name(row, kind))
            args = train_args(root, root / 'run', 1, '--mode', 'joint', '--trainable-prefix', 'weight',
                              '--negative-mode', 'synthetic', '--placement-seed', 7)
            with mock.patch('fragment_ft.training.place_packet', side_effect=lambda packet, *_: packet) as place:
                run_training(FineTuner(backend(), 'joint', ['weight'], hidden=4), rows, root, args, {'test': 'place'})
            self.assertEqual([c[0][2] for c in place.call_args_list],
                             [micro_seed('placement', 42, 7, 0, micro, 0) for micro in range(2)])

    def test_distributed_validation_is_sharded_and_gathered_in_order(self):
        import torch
        import torch.distributed as dist
        import torch.multiprocessing as mp
        from fragment_ft.training import FineTuner, load_delta, predict
        if not dist.is_available():
            self.skipTest('torch.distributed is unavailable')
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0)); port = probe.getsockname()[1]
        with tempfile.TemporaryDirectory() as tmp:
            data.write_manifest(Path(tmp) / 'rows.csv', corpus(tmp))
            with redirect_stdout(io.StringIO()):
                mp.spawn(distributed_worker, args=(tmp, port), nprocs=2)
            report = json.loads((Path(tmp) / 'run/step_000002.json').read_text())
            losses = [x for rank in range(2) for x in json.loads((Path(tmp) / f'losses_{rank}.json').read_text())]
            self.assertEqual(len(losses), 8)
            self.assertAlmostEqual(report['mean_train_loss_since_last_eval'], sum(losses) / 8)
            validation = data.training_rows(data.read_manifest(Path(tmp) / 'rows.csv'))[1]
            self.assertEqual([p['sample_id'] for p in report['predictions']], [r['sample_id'] for r in validation])
            torch.manual_seed(0)
            model = FineTuner(backend(), hidden=4)
            load_delta(model, torch.load(Path(tmp) / 'run/step_000002.pt', weights_only=True))
            expected = predict(model, validation, tmp, torch.device('cpu'))
            for got, want in zip(report['predictions'], expected):
                self.assertAlmostEqual(got['probability'], want['probability'], places=6)


if __name__ == '__main__':
    unittest.main()
