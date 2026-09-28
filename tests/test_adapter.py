"""Adapter logic that runs without Protenix: fake sources, fake upstream modules, small fixtures."""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from fragment_ft import protenix_adapter as adapter
from fragment_ft.data import binding_key
from fragment_ft.synthetic import residue_mapping


def row(sid, target, ligand, smiles, label, split, structure=''):
    return {'sample_id': sid, 'target_id': target, 'ligand_id': ligand, 'smiles': smiles,
            'chem_group': f'G{ligand}', 'label': label, 'split': split, 'structure_id': structure,
            'protein_chain_id': 'P' if structure else '', 'ligand_chain_id': 'L' if structure else ''}


def modules(**named):
    return {name: types.SimpleNamespace(**attributes) for name, attributes in named.items()}


def fake_upstream(data_configs):
    """configs.* and protenix.config.config like upstream: the schema is the base dict it receives."""
    class Attr(dict):
        __getattr__ = dict.__getitem__
    def flat(values, prefix=''):
        for key, value in values.items():
            name = f'{prefix}{key}'
            yield from flat(value, f'{name}.') if isinstance(value, dict) else [(name, value)]
    def wrap(values):
        return Attr({k: wrap(v) if isinstance(v, dict) else v for k, v in values.items()})
    class ConfigManager:
        def __init__(self, base, fill_required_with_null=False):
            self.base, self.config_infos = base, dict(flat(base))
        def merge_configs(self, flattened):
            merged = copy.deepcopy(self.base)
            for name, text in flattened.items():
                *parents, leaf = name.split('.')
                node = merged
                for part in parents:
                    node = node[part]
                node[leaf] = {'True': True, 'False': False}.get(text, text)
            return wrap(merged)
    configs = {'model_name': '', 'train_confidence_only': False, 'triangle_attention': 'triattention',
               'triangle_multiplicative': 'cuequivariance', 'enable_diffusion_shared_vars_cache': True,
               'enable_efficient_fusion': True, 'diffusion_batch_size': 48,
               'model': {'N_cycle': 10}, 'sample_diffusion': {'N_sample': 5}}
    return modules(**{'configs': {}, 'configs.configs_base': {'configs': configs},
                      'configs.configs_data': {'data_configs': data_configs},
                      'configs.configs_inference': {'inference_configs': {}},
                      'configs.configs_model_type': {'model_configs': {'protenix_base_default_v1.0.0': {},
                                                                       'protenix-v2': {}}},
                      'protenix': {}, 'protenix.config': {},
                      'protenix.config.config': {'ConfigManager': ConfigManager}})


def fake_source(root, package=True):
    for name in ('protenix/model/protenix.py', 'configs/configs_base.py') + (('protenix/__init__.py',) if package else ()):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text('')
    return root


def git(root, *args):
    return subprocess.run(['git', '-C', str(root), '-c', 'user.name=t', '-c', 'user.email=t@example.org',
                           '-c', 'commit.gpgsign=false', *args],
                          check=True, capture_output=True, text=True).stdout.strip()


class AdapterLogicTests(unittest.TestCase):
    def test_apo_mapping_is_injective_and_integral(self):
        self.assertEqual(residue_mapping([1, 2], {'residue_offset': 100}), {1: 101, 2: 102})
        self.assertEqual(residue_mapping([1, 2], {'residue_offset': 100, 'residue_map': {'2': 5}}), {1: 101, 2: 5})
        for settings in ({'residue_map': {'2': 1}}, {'residue_offset': 1.5}, {'residue_map': {'3': 9}},
                         {'residue_map': {'1': 1.2}}, {'residue_map': {'1': True}}):
            with self.assertRaises(ValueError): residue_mapping([1, 2], settings)

    def test_strict_deep_update_names_unknown_nested_key(self):
        base = {'a': {'b': 1}}
        self.assertEqual(adapter.deep_update(copy.deepcopy(base), {'a': {'c': 2}}), {'a': {'b': 1, 'c': 2}})
        self.assertEqual(adapter.deep_update(copy.deepcopy(base), {'a': {'b': 3}}, strict=True), {'a': {'b': 3}})
        with self.assertRaisesRegex(ValueError, r'Unknown upstream configuration key: a\.c'):
            adapter.deep_update(copy.deepcopy(base), {'a': {'c': 2}}, strict=True)

    def test_build_config_requires_upstream_to_define_runtime_and_safety_keys(self):
        with mock.patch.dict(sys.modules, fake_upstream({'template': {'fetch_remote': True}})):
            config = adapter.build_config('protenix-v2', {'model': {'N_cycle': 2}})
            self.assertIs(config.data.template['fetch_remote'], False)
            self.assertEqual((config.triangle_attention, config.model.N_cycle), ('torch', '2'))
            with self.assertRaisesRegex(ValueError, 'remote template fetching'):
                adapter.build_config('protenix-v2', {'data': {'template': {'fetch_remote': True}}})
            with self.assertRaisesRegex(ValueError, 'Unknown upstream configuration key: nope'):
                adapter.build_config('protenix-v2', {'nope': 1})
        # A renamed upstream switch must not be shadowed by the adapter's own False.
        with mock.patch.dict(sys.modules, fake_upstream({'template': {'remote_download': True}})):
            with self.assertRaisesRegex(ValueError, r'Unknown upstream configuration key: data\.template\.fetch_remote'):
                adapter.build_config('protenix-v2')

    def connect(self, source, path=()):
        # Hermetic: only the fake checkouts are importable, even where a real protenix is installed.
        stderr = io.StringIO()
        with mock.patch.object(sys, 'path', [*map(str, path)]), mock.patch.dict(os.environ), \
                mock.patch.dict(sys.modules), contextlib.redirect_stderr(stderr):
            sys.modules.pop('protenix', None)
            return adapter.connect_source(source), stderr.getvalue()

    @unittest.skipUnless(shutil.which('git'), 'git is not installed')
    def test_connect_source_records_commit_only_for_its_own_clean_or_dirty_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fake_source(Path(tmp, 'Protenix'))
            git(root, 'init', '-q'); git(root, 'add', '-A'); git(root, 'commit', '-qm', 'x')
            head = git(root, 'rev-parse', 'HEAD')
            with mock.patch.object(adapter, 'UPSTREAM_COMMIT', head):
                result, warning = self.connect(root)
                self.assertEqual((result['actual_source'], result['actual_commit'], result['dirty'], warning),
                                 (str(root.resolve()), head, False, ''))
                (root / 'protenix/model/protenix.py').write_text('changed = 1\n')
                result, warning = self.connect(root)
                self.assertEqual((result['actual_commit'], result['dirty']), (head, True))
                self.assertIn('local changes', warning)
            result, warning = self.connect(root)
            self.assertIn(f'commit {head}', warning)

    @unittest.skipUnless(shutil.which('git'), 'git is not installed')
    def test_connect_source_ignores_enclosing_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            outer = Path(tmp, 'outer')
            root = fake_source(outer / 'vendor/Protenix')
            git(outer, 'init', '-q'); git(outer, 'add', '-A'); git(outer, 'commit', '-qm', 'x')
            result, warning = self.connect(root)
            self.assertEqual((result['actual_commit'], result['dirty']), (None, None))
            self.assertIn('unknown commit', warning)

    def test_connect_source_without_git_records_unknown_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fake_source(Path(tmp, 'Protenix'))
            with mock.patch.object(adapter.subprocess, 'run', side_effect=FileNotFoundError('git')):
                result, warning = self.connect(root)
            self.assertEqual((result['actual_source'], result['actual_commit'], result['dirty']),
                             (str(root.resolve()), None, None))
            self.assertIn('unknown commit', warning)

    def test_connect_source_refuses_protenix_resolved_elsewhere(self):
        with tempfile.TemporaryDirectory() as tmp:
            namespace = fake_source(Path(tmp, 'ns'), package=False)
            with self.assertRaisesRegex(ValueError, 'not a regular package'):
                self.connect(namespace)
            installed = fake_source(Path(tmp, 'installed'))
            # A regular package later on sys.path wins over a namespace directory earlier on it.
            with self.assertRaisesRegex(ValueError, 'not --protenix-source'):
                self.connect(namespace, path=[installed])

    def test_match_interface_needs_exactly_one_protein_ligand_record(self):
        target = row('p1', 'A', 'L1', 'CCO', '1', 'train', 'S1')
        interface = {'pdb_id': 'S1', 'type': 'interface', 'chain_1_id': 'L', 'mol_1_type': 'ligand',
                     'chain_2_id': 'P', 'mol_2_type': 'protein'}
        other = [dict(interface, type='chain'), dict(interface, pdb_id='S2'), dict(interface, chain_1_id='Q')]
        self.assertEqual(adapter.match_interface([*other, interface], target), 3)
        for records in (other, [interface, interface]):
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                adapter.match_interface(records, target)

    def test_prepare_runs_cheap_checks_before_creating_output(self):
        rows = [row('p1', 'A', 'L1', 'CCO', '1', 'train'), row('n1', 'A', 'L2', 'CCN', '0', 'train'),
                row('v1', 'B', 'L3', 'CCC', '0', 'val')]
        targets = {'A': {'sequence': 'ACD'}, 'B': {'sequence': 'MKV'}}
        identity = mock.Mock(return_value={'checked': True, 'molecules': 3})
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(adapter, 'molecule_identity_check', identity), \
                mock.patch.dict(sys.modules, {'protenix': None, 'torch': None}):
            out = Path(tmp, 'prepared')
            cases = [({'indices': 'idx.csv'}, rows, targets, 'bioassembly-dir'),
                     ({}, rows, {'A': targets['A']}, 'Missing target definition: B'),
                     ({'synthetic': True}, rows, targets, 'A: synthetic preparation requires apo')]
            for kwargs, case_rows, case_targets, message in cases:
                with self.assertRaisesRegex(ValueError, message):
                    adapter.prepare(case_rows, case_targets, out, types.SimpleNamespace(), **kwargs)
                self.assertFalse(out.exists())
            identity.side_effect = ValueError('Molecule X spans chem_groups')
            with self.assertRaisesRegex(ValueError, 'spans chem_groups'):
                adapter.prepare(rows, targets, out, types.SimpleNamespace())
            self.assertEqual(identity.call_args, mock.call(rows, required=True))
            identity.side_effect = None
            # Held-out negatives need no apo: the checks pass and only the (blocked) native import fails.
            with self.assertRaises(ImportError):
                adapter.prepare(rows[:1] + rows[2:], targets, out, types.SimpleNamespace(), synthetic=True)
            self.assertFalse(out.exists())
            out.mkdir()
            with self.assertRaisesRegex(FileExistsError, 'Refusing to overwrite prepared data'):
                adapter.prepare(rows, targets, out, types.SimpleNamespace())


HAS_TORCH = all(importlib.util.find_spec(name) for name in ('torch', 'numpy'))


@unittest.skipUnless(HAS_TORCH, 'PyTorch/numpy are not installed in this Python environment')
class PreparedPacketTests(unittest.TestCase):
    def test_tensor_tree_converts_numpy_leaves_inside_object_arrays(self):
        import numpy as np
        import torch
        leaves = np.empty(2, dtype=object)
        leaves[:] = [np.float32(1.5), np.arange(2)]
        tree = adapter.tensor_tree({'o': leaves, 's': np.array(['a']), 'b': np.array([b'x'])})
        stream = io.BytesIO()
        torch.save(tree, stream)
        stream.seek(0)
        loaded = torch.load(stream, weights_only=True)
        self.assertEqual((loaded['o'][0], loaded['o'][1].tolist(), loaded['s'], loaded['b']), (1.5, [0, 1], ['a'], [b'x']))

    def prepare(self, rows, targets, out, overrides):
        import numpy as np
        import torch
        import fragment_ft.training  # noqa: F401  (import before sys.modules is patched and restored)
        featurized, processed = [], []
        class InferenceDataset:
            def __init__(self, config):
                self.names = [item['name'] for item in json.loads(Path(config.input_json_path).read_text())]
            def __getitem__(self, index):
                featurized.append(self.names[index])
                return {'input_feature_dict': {'x': torch.rand(2), 'meta': np.array([np.float32(1.5)], dtype=object)}}, 'atoms', None
        class Records(list):
            def to_dict(self, orient):
                return list(self)
        class BaseSingleDataset:
            def __init__(self, **kwargs):
                self.indices_list = Records([
                    {'pdb_id': 'S1', 'type': 'interface', 'chain_1_id': 'P', 'mol_1_type': 'protein',
                     'chain_2_id': 'L', 'mol_2_type': 'ligand'}])
            def process_one(self, index, return_atom_token_array=False):
                processed.append(index)
                return {'cropped_atom_array': types.SimpleNamespace(chain_id=['P', 'L']),
                        **{key: {'x': torch.rand(2)} for key in ('input_feature_dict', 'label_dict', 'label_full_dict')}}
        fakes = modules(**{'protenix': {}, 'protenix.data': {}, 'protenix.data.inference': {},
                           'protenix.data.inference.infer_dataloader': {'InferenceDataset': InferenceDataset},
                           'protenix.data.pipeline': {},
                           'protenix.data.pipeline.dataset': {'BaseSingleDataset': BaseSingleDataset}})
        identity, apo = mock.Mock(), mock.Mock()
        synthetic = mock.Mock(return_value={'apo_ca_coverage': 1.0, 'label_dict': {'coordinate': torch.zeros(1, 3)}})
        with mock.patch.dict(sys.modules, fakes), mock.patch.object(adapter, 'molecule_identity_check', identity), \
                mock.patch.object(adapter, 'apo_lookup', apo), \
                mock.patch.object(adapter, 'native_synthetic_packet', synthetic):
            adapter.prepare(rows, targets, out, types.SimpleNamespace(model_name='m'), 'idx.csv', 'bio', 'cif',
                            synthetic=True, provenance={'actual_commit': None}, overrides=overrides)
        self.assertEqual(identity.call_args, mock.call(rows, required=True))
        self.assertEqual(apo.call_args_list, [mock.call('A', targets['A'])])
        self.assertEqual([call.args[0]['sample_id'] for call in synthetic.call_args_list], ['n1'])
        self.assertEqual(processed, [0])
        return featurized, json.loads((out / 'preparation.json').read_text())

    def test_prepare_dedupes_binding_seeds_by_content_and_labels_only_train_rows(self):
        import torch
        rows = [row('p1', 'A', 'L1', 'CCO', '1', 'train', 'S1'), row('p2', 'A', 'L1', 'CCO', '1', 'train'),
                row('n1', 'A', 'L2', 'CCN', '0', 'train'), row('v1', 'B', 'L3', 'CCC', '0', 'val'),
                row('v2', 'B', 'L4', 'CCS', '1', 'val', 'S2')]
        # C is never referenced, so its MSA must not switch MSA on for the prepared rows.
        targets = {'A': {'sequence': 'ACD'}, 'B': {'sequence': 'MKV'}, 'C': {'sequence': 'GG', 'pairedMsaPath': 'm.a3m'}}
        overrides = {'data': {'template': {'fetch_remote': False}}}
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp, 'first'), Path(tmp, 'second')
            featurized, preparation = self.prepare(rows, targets, first, overrides)
            self.assertEqual(featurized, ['p1', 'n1', 'v1', 'v2'])
            self.assertTrue(os.path.samefile(first / 'p1.binding.pt', first / 'p2.binding.pt'))
            packet = torch.load(first / 'p2.binding.pt', weights_only=True)
            self.assertEqual((packet['kind'], packet['binding_key'], 'row_hash' in packet),
                             ('binding', binding_key(rows[1]), False))
            self.assertEqual(packet['input_feature_dict']['meta'], [1.5])
            self.assertEqual(sorted(p.name for p in first.glob('*.structure.pt')), ['p1.structure.pt'])
            self.assertEqual(sorted(p.name for p in first.glob('*.synthetic.pt')), ['n1.synthetic.pt'])
            self.assertEqual(torch.load(first / 'n1.synthetic.pt', weights_only=True)['kind'], 'synthetic')
            self.assertEqual((preparation['upstream_overrides'], preparation['binding_msa'],
                              preparation['binding_samples'], preparation['binding_featurizations'],
                              preparation['positive_structures']), (overrides, False, 5, 4, 1))
            # Row order and the rest of the process RNG do not change any prepared tensor.
            torch.rand(5)
            featurized, again = self.prepare(rows[::-1], targets, second, overrides)
            self.assertEqual(featurized, ['v2', 'v1', 'n1', 'p2'])
            self.assertEqual(again['artifacts'], preparation['artifacts'])


if __name__ == '__main__':
    unittest.main()
