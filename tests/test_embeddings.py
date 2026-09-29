"""Disk embeddings and CPU head workflows, using a tiny real Torch backend."""
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from fragment_ft import data
from fragment_ft.__main__ import main
from test_training import backend, corpus


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch is not installed')
class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.features = self.root / 'features'
        self.features.mkdir()
        self.rows = corpus(self.features)[:8]
        self.manifest = self.root / 'manifest.csv'
        data.write_manifest(self.manifest, self.rows)
        self.rows = data.read_manifest(self.manifest)
        self.preparation = {'model_name': 'protenix_base_default_v1.0.0', 'upstream_overrides': {},
                            'source': {'actual_commit': 'mock'},
                            'artifacts': {p.name: data.file_hash(p) for p in self.features.glob('*.pt')}}
        data.write_json(self.features / 'preparation.json', self.preparation)
        self.identity = {'model_name': self.preparation['model_name'], 'upstream_overrides': {},
                         'source': self.preparation['source'], 'base_checkpoint_sha256': 'a' * 64}
        self.backend = backend()
        self.cpu = torch.device('cpu')

    def extract(self, name='embeddings', **kwargs):
        from fragment_ft.embeddings import extract_embeddings
        path = self.root / name
        extract_embeddings(self.backend, self.rows, self.features, path, self.identity, self.cpu, **kwargs)
        return path

    def cli(self, *args):
        err = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(err):
            result = main([str(a) for a in args])
        self.assertEqual(result, 0, err.getvalue())

    def test_unique_encoding_determinism_and_fresh_model_disk_reuse(self):
        import torch
        from fragment_ft.embeddings import CachedBackend, DiskEmbeddings
        from fragment_ft.training import FineTuner, predict
        self.rows.append(dict(self.rows[0], label='1', split='val'))
        with mock.patch.object(self.backend, 'encode', wraps=self.backend.encode) as encode:
            first = self.extract()
            self.assertEqual(encode.call_count, 8)
        second = self.extract('repeat')
        one, two = DiskEmbeddings([first], self.rows), DiskEmbeddings([second], self.rows)
        for key in one.entries:
            self.assertTrue(torch.equal(one[key], two[key]))
        native = FineTuner(self.backend, hidden=4)
        cached = FineTuner(CachedBackend(one.identity), hidden=4, embeddings=one)
        cached.head.load_state_dict(native.head.state_dict())
        expected = predict(native, self.rows, self.features, self.cpu)
        with mock.patch('fragment_ft.training.load_packet', side_effect=AssertionError('packet read')), \
                mock.patch('torch.load', side_effect=AssertionError('per-forward disk load')), \
                mock.patch('fragment_ft.embeddings.file_hash', side_effect=AssertionError('per-forward hash')):
            self.assertEqual(predict(cached, self.rows, None, self.cpu), expected)

    def test_shards_cover_keys_and_reject_missing_duplicate_incompatible(self):
        from fragment_ft.embeddings import DiskEmbeddings
        paths = [self.extract(f'shard{i}', shard_index=i, shard_count=2) for i in range(2)]
        self.assertEqual(len(DiskEmbeddings(paths, self.rows).entries), 8)
        with self.assertRaisesRegex(ValueError, 'Missing embeddings'):
            DiskEmbeddings(paths[:1], self.rows)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            DiskEmbeddings(paths + paths[:1], self.rows)
        self.identity['base_checkpoint_sha256'] = 'b' * 64
        other = self.extract('other', shard_index=1, shard_count=2)
        with self.assertRaisesRegex(ValueError, 'Incompatible'):
            DiskEmbeddings([paths[0], other], self.rows)

    def test_stale_features_and_corruption_are_rejected(self):
        import torch
        from fragment_ft.embeddings import DiskEmbeddings
        path = self.extract()
        packet = self.features / data.packet_name(self.rows[0], 'binding')
        value = torch.load(packet, weights_only=True)
        value['input_feature_dict']['x'] += 1
        torch.save(value, packet)
        self.preparation['artifacts'][packet.name] = data.file_hash(packet)
        (self.features / 'preparation.json').unlink()
        data.write_json(self.features / 'preparation.json', self.preparation)
        with self.assertRaisesRegex(ValueError, 'Stale embedding'):
            DiskEmbeddings([path], self.rows, self.features)
        other = self.extract('changed')
        with self.assertRaisesRegex(ValueError, 'feature provenance'):
            DiskEmbeddings([path, other], self.rows)
        vector = path / (data.binding_key(self.rows[0]) + '.pt')
        vector.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'Corrupt embedding'):
            DiskEmbeddings([path], self.rows)

    def test_metadata_corruption_and_invalid_shards_are_rejected(self):
        from fragment_ft.embeddings import DiskEmbeddings
        with self.assertRaises(ValueError):
            self.extract(shard_index=2, shard_count=2)
        path = self.extract()
        metadata = data.read_json(path / 'embeddings.json')
        metadata['identity']['c_s'] += 1
        (path / 'embeddings.json').unlink()
        data.write_json(path / 'embeddings.json', metadata)
        with self.assertRaisesRegex(ValueError, 'Corrupt embedding metadata'):
            DiskEmbeddings([path], self.rows)

    def test_cpu_cli_train_predict_resume_init_without_native_or_base(self):
        import torch
        paths = [self.extract(f'shard{i}', shard_index=i, shard_count=2) for i in range(2)]
        common = ['--embeddings', *paths, '--device', 'cpu', '--precision', 'fp32']
        train = ['train', self.manifest, *common, '--steps', '2', '--eval-every', '1',
                 '--accumulate', '1', '--hidden', '4']
        with mock.patch.dict('sys.modules', {'fragment_ft.protenix_adapter': None}), \
                mock.patch('fragment_ft.training.load_packet', side_effect=AssertionError('packet read')):
            self.cli(*train, '--output', self.root / 'run')
            checkpoint = self.root / 'run/step_000002.pt'
            self.cli(*train, '--output', self.root / 'resumed', '--steps', '3', '--resume', checkpoint)
            self.cli(*train, '--output', self.root / 'initialized', '--init-checkpoint', checkpoint)
            self.cli('predict', self.manifest, *common, '--checkpoint', checkpoint,
                     '--split', 'val', '--output', self.root / 'predictions.json')
        report = data.read_json(self.root / 'predictions.json')
        self.assertEqual(len(report['predictions']), 4)
        payload = torch.load(checkpoint, weights_only=True)
        self.assertTrue(all(name.startswith('head.') for name in payload['delta']))
        self.assertEqual(payload['metadata']['base_checkpoint_sha256'], self.identity['base_checkpoint_sha256'])
        self.assertEqual(len(payload['metadata']['embedding_features']), 8)

    def test_embed_cli_and_cached_mode_restrictions(self):
        base = self.root / 'base.pt'
        base.write_bytes(b'mock checkpoint')
        with mock.patch('fragment_ft.protenix_adapter.connect_source', return_value=self.identity['source']), \
                mock.patch('fragment_ft.protenix_adapter.build_config', return_value={}), \
                mock.patch('fragment_ft.protenix_adapter.native_backend', return_value=self.backend):
            self.cli('embed', self.manifest, '--features', self.features, '--base-checkpoint', base,
                     '--output', self.root / 'cli', '--device', 'cpu', '--shard-index', '0', '--shard-count', '2')
        from fragment_ft.__main__ import check_train_args, parser
        for flags in (['--mode', 'joint', '--trainable-prefix', 'weight'], ['--negative-mode', 'synthetic']):
            args = parser().parse_args(['train', str(self.manifest), '--embeddings', str(self.root / 'cli'),
                                       '--output', str(self.root / 'run'), *flags])
            with self.assertRaisesRegex(ValueError, 'head/classifier'):
                check_train_args(args)
