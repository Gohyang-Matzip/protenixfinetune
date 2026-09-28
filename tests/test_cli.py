"""CLI wiring through main(). Native steps run against stand-ins, so most tests need neither torch nor Protenix."""
from contextlib import redirect_stderr, redirect_stdout
import http.client
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from fragment_ft import data, protenix_adapter
import fragment_ft.sources  # noqa: F401  imported before any sys.modules patch, so patches hit the live module
from fragment_ft.__main__ import main, parser

EXAMPLES = Path(__file__).resolve().parent.parent / 'examples'
MODEL = 'protenix_base_default_v1.0.0'


def run(*argv):
    """Exit code and stderr of main(); stdout is discarded."""
    err = io.StringIO()
    with redirect_stdout(io.StringIO()), redirect_stderr(err):
        code = main([str(a) for a in argv])
    return code, err.getvalue()


def screen(root, name='manifest.csv', drop=()):
    """Two targets, chemistry split, plus a test-only task no checkpoint trains and a phenotypic row."""
    rows = [dict(sample_id=f'{t}_{i}', target_id=t, target_group=t, ligand_id=f'L{i}', smiles='C' * (i + 1),
                 chem_group=f'G{i}', label=str(i % 2), split=('train', 'train', 'val', 'val', 'test', 'test')[i],
                 assay_id=f'{t}_screen', assay_type='xray', endpoint='hit', campaign_id='study',
                 source='example', source_url='https://example.org/study', quality='pass')
            for t in ('A', 'B') for i in range(6)]
    rows += [dict(rows[4], sample_id='A_enzyme', assay_id='enzyme', assay_type='biochemical', endpoint='inhibition'),
             dict(rows[4], sample_id='A_cell', assay_id='cell', assay_type='phenotypic', endpoint='viability')]
    path = Path(root) / name
    data.write_manifest(path, [data.compact(r) for r in rows if r['sample_id'] not in drop])
    return path, data.read_manifest(path)


def features(root, rows, save=lambda row, path: path.write_bytes(b'packet')):
    directory = Path(root) / 'features'
    directory.mkdir()
    for row in rows:
        if data.predictable(row):
            save(row, directory / data.packet_name(row, 'binding'))
    data.write_json(directory / 'preparation.json', {
        'model_name': MODEL, 'upstream_overrides': {}, 'source': {'actual_commit': 'abc'},
        'artifacts': {p.name: data.file_hash(p) for p in sorted(directory.glob('*.pt'))}})
    return directory


def stubs(payload=None):
    """Stand-ins for torch, the Protenix adapter and the training loop; the mocks record what ran."""
    torch, adapter, training = mock.MagicMock(), mock.MagicMock(), mock.MagicMock()
    torch.load.return_value = payload
    adapter.connect_source.return_value = {'actual_commit': 'abc'}
    training.predict.side_effect = lambda model, rows, *rest: [
        dict(r, probability=.9 if r['label'] == '1' else .1, task=data.task_name(r)) for r in rows]
    patcher = mock.patch.dict(sys.modules, {'torch': torch, 'fragment_ft.protenix_adapter': adapter,
                                            'fragment_ft.training': training})
    return patcher, torch, adapter, training


class DataCommandTests(unittest.TestCase):
    def test_documented_example_chain_refuses_overwrite_and_audits_cross_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            steps = [('import-csv', EXAMPLES/'general_screen.csv', '--mapping', EXAMPLES/'general_mapping.json',
                      '--output', root/'import'),
                     ('split', root/'import/manifest.csv', '--strategy', 'both', '--seed', 42, '--output', root/'both.csv'),
                     ('audit', root/'both.csv', '--output', root/'audit.json'),
                     ('inputs', root/'both.csv', '--targets', EXAMPLES/'general_targets.json', '--output', root/'inputs.json')]
            for step in steps:
                self.assertEqual(run(*step)[0], 0, step[0])
            provenance = json.loads((root/'import/provenance.json').read_text())
            self.assertEqual(provenance['mapping_sha256'], data.file_hash(EXAMPLES/'general_mapping.json'))
            audit = json.loads((root/'audit.json').read_text())
            # The numbers README and docs/VALIDATION.md publish for this chain.
            self.assertEqual({name: (s['observations'], s['trainable']) for name, s in audit['splits'].items()},
                             {'train': (67, 64), 'val': (4, 4), 'test': (4, 4), 'excluded': (72, 0)})
            self.assertEqual(set(audit['molecule_identity']), {'checked', 'molecules'})
            rows = data.read_manifest(root/'both.csv')
            names = {item['name'] for item in json.loads((root/'inputs.json').read_text())}
            self.assertEqual(len(names), 74)
            self.assertFalse(names & {r['sample_id'] for r in rows if not data.predictable(r)})
            before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
            for step in steps:
                self.assertEqual(run(*step)[0], 2, f'{step[0]} must refuse to overwrite')
            self.assertEqual(before, {p: p.read_bytes() for p in root.rglob('*') if p.is_file()})

            data.write_manifest(root/'pretrain.csv', [dict(r, split='train') for r in rows])
            self.assertEqual(run('audit', root/'both.csv', '--against', root/'pretrain.csv', '--output', root/'x.json')[0], 0)
            self.assertTrue(json.loads((root/'x.json').read_text())['cross_corpus']['leakage'])
            self.assertEqual(run('audit', root/'both.csv', '--against', root/'both.csv', '--output', root/'self.json')[0], 0)
            self.assertFalse(json.loads((root/'self.json').read_text())['cross_corpus']['leakage'])

    def test_inputs_resolve_target_paths_and_drop_phenotypes_and_apo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, rows = screen(root)
            context = root/'context'
            (context/'msa_A').mkdir(parents=True)
            (context/'A.a3m').write_text('>A\nACDE\n'); (context/'apo.cif').write_text('')
            targets = {'A': {'sequence': 'ACDE', 'pairedMsaPath': 'A.a3m', 'msa': {'precomputed_msa_dir': 'msa_A'},
                             'apo': {'path': 'apo.cif', 'chain_id': 'A'}},
                       'B': {'sequence': 'FGHI'}}
            (context/'targets.json').write_text(json.dumps(targets))
            self.assertEqual(run('inputs', manifest, '--targets', context/'targets.json', '--output', root/'in.json')[0], 0)
            inputs = json.loads((root/'in.json').read_text())
            self.assertEqual({i['name'] for i in inputs}, {r['sample_id'] for r in rows if data.predictable(r)})
            self.assertNotIn('A_cell', {i['name'] for i in inputs})
            self.assertFalse(any('apo' in s['proteinChain'] for i in inputs for s in i['sequences'] if 'proteinChain' in s))
            protein = next(i for i in inputs if i['name'] == 'A_0')['sequences'][0]['proteinChain']
            self.assertEqual(protein['pairedMsaPath'], str((context/'A.a3m').resolve()))
            self.assertEqual(protein['msa']['precomputed_msa_dir'], str((context/'msa_A').resolve()))
            for bad, message in (([targets['A']], 'must map'), ({'A': 'ACDE', 'B': 'FGHI'}, 'must map'),
                                 (dict(targets, A=dict(targets['A'], msa={'precomputed_msa_dir': 'missing'})), 'Missing'),
                                 (dict(targets, A=dict(targets['A'], msa={'precomputed_msa_dir': 5})), 'must be a string'),
                                 (dict(targets, A=dict(targets['A'], msa='msa_A')), 'msa must be an object'),
                                 (dict(targets, A=dict(targets['A'], pairedMsaPath=5)), 'must be a string'),
                                 (dict(targets, A=dict(targets['A'], sequence=123)), 'sequence must be a string'),
                                 (dict(targets, A=dict(targets['A'], apo={'path': 5})), 'must be a string'),
                                 (dict(targets, A=dict(targets['A'], apo={'chain_id': 'A'})), 'apo must be an object')):
                (context/'bad.json').write_text(json.dumps(bad))
                code, err = run('inputs', manifest, '--targets', context/'bad.json', '--output', root/'bad.json')
                self.assertEqual(code, 2, bad); self.assertRegex(err, f'(?m)^error:.*{message}')
            (context/'bad.json').write_text('{"A": {"sequence": "ACDE"}, "A": {"sequence": "FGHI"}}')
            self.assertIn('duplicate', run('inputs', manifest, '--targets', context/'bad.json', '--output', root/'bad.json')[1])
            self.assertFalse((root/'bad.json').exists())

    def test_duplicate_mapping_keys_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text = (EXAMPLES/'general_mapping.json').read_text().replace('"Inactive": "0",', '"Inactive": "uncertain", "Inactive": "0",')
            (root/'mapping.json').write_text(text)
            code, err = run('import-csv', EXAMPLES/'general_screen.csv', '--mapping', root/'mapping.json', '--output', root/'out')
            self.assertEqual(code, 2); self.assertIn('duplicate JSON keys', err)
            self.assertFalse((root/'out').exists())

    def test_merge_refuses_mixed_split_state_and_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, rows = screen(root)
            data.write_manifest(root/'unsplit.csv', [dict(r, sample_id='u' + r['sample_id'], split='') for r in rows])
            code, err = run('merge', manifest, root/'unsplit.csv', '--output', root/'mixed.csv')
            self.assertEqual(code, 2); self.assertIn('all-split', err)
            self.assertFalse((root/'mixed.csv').exists())
            self.assertEqual(run('merge', manifest, '--output', root/'merged.csv')[0], 0)
            first = (root/'merged.csv').read_bytes()
            self.assertEqual(run('merge', manifest, '--output', root/'merged.csv')[0], 2)
            self.assertEqual((root/'merged.csv').read_bytes(), first)

    def test_compare_validates_records_before_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, rows = screen(root)
            good = [dict(r, probability=.9 if r['label'] == '1' else .1, task=data.task_name(r))
                    for r in rows if data.trainable(r) and r['split'] == 'test']
            cases = {'list': good, 'labels': {'predictions': [dict(r, label=int(r['label'])) for r in good]},
                     # Only some rows bad: the rest would still score, so the bad ones must not be dropped silently.
                     'some_int_labels': {'predictions': [dict(good[0], label=int(good[0]['label']))] + good[1:]},
                     'unknown_label': {'predictions': [dict(good[0], label='2')] + good[1:]},
                     'int_target': {'predictions': [dict(good[0], target_id=5)] + good[1:]},
                     'training': {'predictions': good, 'training': ['x']},
                     'probabilities': {'predictions': [dict(r, probability=str(r['probability'])) for r in good]},
                     'unscorable': {'predictions': [dict(r, label='uncertain') for r in good]}}
            for name, report in cases.items():
                (root/f'{name}.json').write_text(json.dumps(report))
                code, err = run('compare', root/f'{name}.json', '--output', root/'out.json')
                self.assertEqual(code, 2, name); self.assertRegex(err, '(?m)^error:')
            self.assertFalse((root/'out.json').exists())
            for name, trained_on, same in (('a', 'x', True), ('b', 'y', False)):
                (root/f'{name}.json').write_text(json.dumps({'predictions': good, 'same_manifest_as_training': same,
                                                            'training': {'manifest_sha256': trained_on}}))
            code, err = run('compare', root/'a.json', root/'b.json', '--output', root/'out.json')
            self.assertEqual(code, 0); self.assertIn('different manifests', err)
            self.assertIn(f'warning: {root/"b.json"} evaluated a manifest other than', err)
            self.assertNotIn(str(root/'a.json'), err)
            out = json.loads((root/'out.json').read_text())
            self.assertEqual([e['same_manifest_as_training'] for e in out.values()], [True, False])

    def test_counts_are_positive_integers_and_new_flags_default(self):
        train = ['train', 'm.csv', '--features', 'f', '--base-checkpoint', 'b', '--output', 'o']
        for argv in (['compare', 'x.json', '--output', 'y', '--top-k', '0'],
                     ['predict', 'm.csv', '--features', 'f', '--checkpoint', 'c', '--base-checkpoint', 'b',
                      '--output', 'o', '--top-k', '-1'],
                     train + ['--steps', '0'], train + ['--accumulate', '1.5'], train + ['--dist-timeout-minutes', '0'],
                     train + ['--top-k', '0'],
                     ['fetch', 'https://example.org', '--output', 'o', '--max-mb', '0']):
            with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
                parser().parse_args(argv)
        args = parser().parse_args(train)
        self.assertEqual((args.select_metric, args.dist_timeout_minutes), ('average_precision', 60))

    def test_import_lit_records_the_mapping_hash(self):
        mapping = EXAMPLES/'general_mapping.json'
        with mock.patch('fragment_ft.sources.import_lit', return_value=[]) as lit:
            self.assertEqual(run('import-lit', '--active', 'a', '--inactive', 'i', '--mapping', mapping,
                                 '--compounds', 'c', '--output', 'o')[0], 0)
        self.assertEqual(lit.call_args.kwargs['mapping_sha256'], data.file_hash(mapping))

    def test_incomplete_download_is_a_clean_error(self):
        with mock.patch('fragment_ft.sources.download', side_effect=http.client.IncompleteRead(b'abc', 10)):
            code, err = run('fetch', 'https://example.org/x', '--output', 'unused')
        self.assertEqual(code, 2); self.assertIn('IncompleteRead', err)


class NativeCommandOrderTests(unittest.TestCase):
    """Refusals must happen before the output exists, the base model loads or inference starts."""

    def test_prepare_checks_everything_cheap_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, rows = screen(root)
            targets = {'A': {'sequence': 'ACDE'}, 'B': {'sequence': 'FGHI'}}
            (root/'targets.json').write_text(json.dumps(targets))
            (root/'only_A.json').write_text(json.dumps({'A': targets['A']}))
            (root/'upstream.json').write_text(json.dumps({'data': {'msa': {'min_size': 1}}}))
            (root/'list.json').write_text('[]')
            (root/'indices.csv').write_text(''); (root/'taken').mkdir()
            prepare = ['prepare', manifest, '--output', root/'prepared', '--targets']
            # The real adapter.prepare makes its own cheap refusals; only Protenix entry points (and RDKit) are stubbed.
            native = dict(connect_source=lambda source: {'actual_commit': 'abc'}, build_config=lambda *a: None,
                          molecule_identity_check=lambda rows, required=False: None)
            with mock.patch.multiple(protenix_adapter, **native):
                for extra, message in (([root/'targets.json', '--positive-indices', root/'indices.csv'], '--bioassembly-dir'),
                                       ([root/'targets.json', '--positive-indices', root/'missing.csv',
                                         '--bioassembly-dir', root, '--mmcif-dir', root], 'Missing structural input'),
                                       ([root/'only_A.json'], 'Missing target definition: B'),
                                       ([root/'targets.json', '--synthetic'], 'apo.path and apo.chain_id'),
                                       ([root/'targets.json', '--crop-size', 0], '--crop-size'),
                                       ([root/'targets.json', '--upstream-config', root/'list.json'], 'JSON object'),
                                       ([root/'targets.json', '--output', root/'taken'], '(?:overwrite|already exists)')):
                    code, err = run(*prepare, *extra)
                    self.assertEqual(code, 2, extra); self.assertRegex(err, f'(?m)^error:.*{message}')
                self.assertFalse((root/'prepared').exists())
                self.assertEqual(list((root/'taken').iterdir()), [])
                with mock.patch.object(protenix_adapter, 'prepare') as adapter_prepare:
                    self.assertEqual(run(*prepare, root/'targets.json', '--upstream-config', root/'upstream.json')[0], 0)
            (call,) = adapter_prepare.call_args_list
            self.assertEqual(call.kwargs['overrides'], {'data': {'msa': {'min_size': 1}}})
            self.assertEqual([r['sample_id'] for r in call.args[0]], [r['sample_id'] for r in rows if data.predictable(r)])

    def test_train_rejects_bad_flags_before_loading_the_base_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, rows = screen(root)
            feats = features(root, rows)
            (root/'base.pt').write_bytes(b'base'); (root/'init.pt').write_bytes(b'init'); (root/'taken').mkdir()
            (root/'upstream.json').write_text(json.dumps({'data': {'msa': {'min_size': 1}}}))
            previous = {'base_checkpoint_sha256': '0' * 64, 'model_name': MODEL, 'upstream_overrides': {}}
            train = ['train', manifest, '--features', feats, '--base-checkpoint', root/'base.pt', '--device', 'cpu', '--output']
            patcher, _, adapter, training = stubs({'metadata': previous})
            with patcher:
                for extra in ([root/'taken'], [root/'new', '--trainable-prefix', 'pairformer_stack'],
                              [root/'new', '--mode', 'joint'], [root/'new', '--init-checkpoint', root/'init.pt'],
                              [root/'new', '--upstream-config', root/'upstream.json']):
                    code, err = run(*train, *extra)
                    self.assertEqual(code, 2, extra); self.assertRegex(err, '(?m)^error:')
                adapter.native_backend.assert_not_called(); training.run_training.assert_not_called()
                self.assertEqual(run(*train, root/'new', '--select-metric', 'log_loss')[0], 0)
                # A cache from before prepare recorded its overrides stays usable, with warnings.
                preparation = json.loads((feats/'preparation.json').read_text())
                del preparation['upstream_overrides']; preparation['source'] = {'actual_commit': 'old'}
                (feats/'preparation.json').write_text(json.dumps(preparation))
                code, err = run(*train, root/'legacy', '--upstream-config', root/'upstream.json')
                self.assertEqual(code, 0); self.assertIn('does not record its upstream overrides', err)
                self.assertIn('warning: Prepared data used Protenix commit old, current is abc', err)
            call = training.run_training.call_args_list[0]
            metadata = call.args[4]
            self.assertEqual(metadata['task_names'], ['xray:hit'])
            self.assertEqual((metadata['seed_scheme'], metadata['select_metric']), (2, 'log_loss'))
            self.assertEqual(metadata['manifest_sha256'], data.file_hash(manifest))

    def test_predict_checks_before_inference_and_skips_untrained_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, rows = screen(root)
            other, _ = screen(root, 'other.csv', drop=('A_cell',))
            feats = features(root, rows)
            (root/'base.pt').write_bytes(b'base'); (root/'ckpt.pt').write_bytes(b'ckpt'); (root/'taken.json').write_text('{}')
            metadata = {'model_name': MODEL, 'upstream_overrides': {}, 'source': {'actual_commit': 'abc'},
                        'base_checkpoint_sha256': data.file_hash(root/'base.pt'), 'manifest_sha256': data.file_hash(manifest),
                        'task_names': ['xray:hit'], 'mode': 'head', 'trainable_prefix': [], 'hidden': 4}
            predict = ['predict', '--features', feats, '--checkpoint', root/'ckpt.pt', '--base-checkpoint', root/'base.pt',
                       '--device', 'cpu']
            patcher, torch, adapter, training = stubs({'metadata': metadata})
            with patcher:
                for extra, changed in (([manifest, '--output', root/'taken.json'], {}),
                                       ([other, '--output', root/'p.json'], {}),
                                       ([manifest, '--output', root/'p.json'], {'task_names': ['direct_binding:binding']}),
                                       ([manifest, '--output', root/'p.json'], {'upstream_overrides': {'data': {'x': 1}}}),
                                       ([manifest, '--output', root/'p.json'], {'model_name': 'protenix-v2'})):
                    torch.load.return_value = {'metadata': dict(metadata, **changed)}
                    code, err = run(*predict, *extra)
                    self.assertEqual(code, 2, (extra, changed)); self.assertRegex(err, '(?m)^error:')
                torch.load.return_value = {'metadata': {k: v for k, v in metadata.items() if k != 'model_name'}}
                self.assertIn('missing key', run(*predict, manifest, '--output', root/'p.json')[1])
                adapter.native_backend.assert_not_called(); training.predict.assert_not_called()
                self.assertFalse((root/'p.json').exists())
                torch.load.return_value = {'metadata': metadata}
                code, err = run(*predict, manifest, '--output', root/'p.json')
                self.assertEqual(code, 0); self.assertIn('never trained', err)
                self.assertEqual(run(*predict, other, '--output', root/'other.json', '--allow-different-manifest')[0], 0)
            report = json.loads((root/'p.json').read_text())
            self.assertEqual(report['untrained_tasks'], {'biochemical:inhibition': 1})
            self.assertTrue(report['same_manifest_as_training'])
            self.assertEqual({(r['task'], r['split']) for r in report['predictions']}, {('xray:hit', 'test')})
            self.assertFalse(json.loads((root/'other.json').read_text())['same_manifest_as_training'])


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch not installed')
class NativeWiringTests(unittest.TestCase):
    def test_train_predict_export_through_main_with_a_tiny_backend(self):
        import torch
        from fragment_ft import protenix_adapter

        class Backend(torch.nn.Module):
            c_s = c_z = 2
            def __init__(self):
                super().__init__(); self.model = torch.nn.Linear(2, 2)
            def encode(self, f):
                s = self.model(f['x']); return s, s[:, None] + s[None, :]

        x = {'x': torch.eye(3, 2), 'atom_to_token_idx': torch.arange(3),
             'is_protein': torch.tensor([1, 1, 0]), 'is_ligand': torch.tensor([0, 0, 1])}
        save = lambda row, path: torch.save({'version': 1, 'kind': 'binding', 'row_hash': data.row_hash(row),
                                             'input_feature_dict': x}, path)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, rows = screen(root)
            feats = features(root, rows, save)
            (root/'base.pt').write_bytes(b'base'); (root/'other.pt').write_bytes(b'other')
            common = ['--features', feats, '--device', 'cpu', '--precision', 'fp32']
            train = ['train', manifest, *common, '--steps', 2, '--eval-every', 1, '--accumulate', 1, '--hidden', 4]
            native = dict(connect_source=lambda source: {'actual_commit': 'abc'}, build_config=lambda *a: None,
                          native_backend=lambda config, checkpoint: Backend())
            with mock.patch.multiple(protenix_adapter, **native):
                self.assertEqual(run(*train, '--base-checkpoint', root/'base.pt', '--output', root/'run')[0], 0)
                checkpoint = root/'run/step_000002.pt'
                for extra in (['--base-checkpoint', root/'other.pt', '--init-checkpoint', checkpoint, '--output', root/'a'],
                              ['--base-checkpoint', root/'base.pt', '--resume', root/'run/step_000001.pt',
                               '--head-lr', .5, '--output', root/'b'],
                              ['--base-checkpoint', root/'base.pt', '--resume', checkpoint, '--output', root/'run']):
                    code, err = run(*train, *extra)
                    self.assertEqual(code, 2, extra); self.assertRegex(err, '(?m)^error:')
                self.assertEqual(run(*train, '--base-checkpoint', root/'base.pt', '--init-checkpoint', checkpoint,
                                     '--output', root/'init')[0], 0)
                predict = ['predict', manifest, *common, '--checkpoint', checkpoint, '--base-checkpoint', root/'base.pt']
                self.assertEqual(run(*predict, '--output', root/'p.json')[0], 0)
                export = ['export', '--checkpoint', checkpoint, '--base-checkpoint', root/'base.pt', '--output', root/'native.pt']
                self.assertEqual(run(*export)[0], 0)
                self.assertEqual(run(*export)[0], 2)
            report = json.loads((root/'p.json').read_text())
            self.assertEqual(report['untrained_tasks'], {'biochemical:inhibition': 1})
            self.assertEqual(len(report['predictions']), 4)
            self.assertEqual(report['training']['task_names'], ['xray:hit'])
            state = torch.load(root/'native.pt', map_location='cpu', weights_only=True)['model']
            self.assertEqual(set(state), set(Backend().model.state_dict()))


if __name__ == '__main__':
    unittest.main()
