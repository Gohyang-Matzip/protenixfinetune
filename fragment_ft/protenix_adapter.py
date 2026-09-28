"""Optional adapter for upstream Protenix commit 4c355be4553512f72453ecbfb65e69f4c35d1413.

No model implementation is vendored and no installation/download is performed.
"""
from collections import defaultdict
import copy
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .data import (binding_key, file_hash, make_inputs, molecule_identity_check, packet_name, row_hash,
                   structure_eligible, synthetic_eligible, write_json)
from .synthetic import apo_lookup, native_synthetic_packet

UPSTREAM_COMMIT = '4c355be4553512f72453ecbfb65e69f4c35d1413'


def connect_source(source=None):
    root = None
    if source:
        root = Path(source).resolve()
        if not (root / 'protenix/model/protenix.py').is_file() or not (root / 'configs/configs_base.py').is_file():
            raise ValueError('--protenix-source must point to an existing upstream source checkout')
        sys.path.insert(0, str(root))
    spec = importlib.util.find_spec('protenix')
    if spec is None:
        raise RuntimeError('Protenix is not available. No installation was attempted. '
                           'Use this command later in a configured training environment '
                           'with --protenix-source /path/to/Protenix.')
    if spec.origin is None:
        raise ValueError(f'protenix is not a regular package (no __init__.py): {list(spec.submodule_search_locations)}')
    actual_root = Path(spec.origin).resolve().parent.parent
    if root and actual_root != root:
        raise ValueError(f'protenix resolves to {actual_root}, not --protenix-source {root}; '
                         'another copy is installed or already imported')
    # Avoid custom extension compilation; CUDA triangle kernels remain configurable.
    os.environ.setdefault('LAYERNORM_TYPE', 'torch')

    def git(*args):
        try:
            result = subprocess.run(['git', '-C', str(actual_root), *args], capture_output=True, text=True, check=False)
        except OSError:  # git only records provenance; running does not need it
            return None
        return result.stdout.strip() if result.returncode == 0 else None
    # git searches parent directories: only a checkout rooted at the source identifies its commit.
    top = git('rev-parse', '--show-toplevel')
    commit = git('rev-parse', 'HEAD') if top and Path(top).resolve() == actual_root else None
    status = git('status', '--porcelain', '--untracked-files=no') if commit else None
    dirty = None if status is None else bool(status)
    if commit != UPSTREAM_COMMIT or dirty:
        state = (f'commit {commit}' + (' with local changes' if dirty else '') if commit
                 else 'an unknown commit (no git checkout rooted there, or git unavailable)')
        print(f'warning: Protenix at {actual_root} is {state}; the adapter targets {UPSTREAM_COMMIT}', file=sys.stderr)
    return {'expected_upstream_commit': UPSTREAM_COMMIT, 'actual_source': str(actual_root),
            'actual_commit': commit, 'dirty': dirty}


def deep_update(destination, updates, strict=False, prefix=''):
    """Nested merge; strict refuses keys the destination (the upstream schema) does not define."""
    for key, value in updates.items():
        if strict and key not in destination:
            raise ValueError(f'Unknown upstream configuration key: {prefix}{key}')
        if isinstance(value, dict) and isinstance(destination.get(key), dict):
            deep_update(destination[key], value, strict, f'{prefix}{key}.')
        else:
            destination[key] = copy.deepcopy(value)
    return destination


def build_config(model_name, overrides=None):
    from configs.configs_base import configs
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from configs.configs_model_type import model_configs
    from protenix.config.config import ConfigManager

    if model_name not in ('protenix_base_default_v1.0.0', 'protenix-v2'):
        raise ValueError('Supported configurations: protenix_base_default_v1.0.0, protenix-v2')
    base = copy.deepcopy({**configs, 'data': data_configs, **inference_configs})
    deep_update(base, model_configs[model_name])
    # Strict: a renamed upstream key must fail here, not leave e.g. remote template fetching on.
    deep_update(base, {'model_name': model_name, 'train_confidence_only': False,
                      'triangle_attention': 'torch', 'triangle_multiplicative': 'torch',
                      'enable_diffusion_shared_vars_cache': False,
                      'enable_efficient_fusion': False, 'diffusion_batch_size': 4,
                      'model': {'N_cycle': 4}, 'sample_diffusion': {'N_sample': 1},
                      'data': {'template': {'fetch_remote': False}}}, strict=True)
    manager = ConfigManager(base, fill_required_with_null=True)
    flattened = {}
    def flatten(values, prefix=''):
        for key, value in values.items():
            name = f'{prefix}.{key}' if prefix else key
            if isinstance(value, dict):
                flatten(value, name)
            else:
                if name not in manager.config_infos:
                    raise ValueError(f'Unknown upstream configuration key: {name}')
                flattened[name] = ('null' if value is None else ','.join(map(str, value))
                                   if isinstance(value, list) else str(value))
    flatten(overrides or {})
    result = manager.merge_configs(flattened)
    if result.train_confidence_only or result.data.template.get('fetch_remote', False):
        raise ValueError('train_confidence_only and remote template fetching must remain disabled')
    return result


def native_backend(config, checkpoint):
    """Construct only when explicitly requested; normal CLI validation never imports torch."""
    import torch
    from protenix.model.loss import ProtenixLoss
    from protenix.model.protenix import Protenix, update_input_feature_dict
    from protenix.utils.permutation.permutation import SymmetricPermutation
    from protenix.utils.torch_utils import autocasting_disable_decorator

    class ProtenixBackend(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Protenix(config)
            self.c_s, self.c_z = config.c_s, config.c_z
            self.criterion = ProtenixLoss(config)
            self.permutation = SymmetricPermutation(config, error_dir=None)
            # weights_only avoids executing arbitrary pickle objects. Legacy checkpoints
            # needing unsafe globals must be converted by the owner in a trusted environment.
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            state = state.get('model', state)
            state = {key.removeprefix('module.'): value for key, value in state.items()}
            self.model.load_state_dict(state, strict=True)

        def encode(self, features):
            features = self.model.relative_position_encoding.generate_relp(dict(features))
            features = update_input_feature_dict(features)
            _, single, pair = self.model.get_pairformer_output(
                input_feature_dict=features, N_cycle=config.model.N_cycle,
                inplace_safe=False, chunk_size=None)
            return single, pair

        def structure_loss(self, batch, step):
            features = dict(batch['input_feature_dict'])
            prediction, labels, _ = self.model(
                input_feature_dict=features,
                label_dict=copy.deepcopy(batch['label_dict']),
                label_full_dict=copy.deepcopy(batch['label_full_dict']),
                mode='train', current_step=step,
                symmetric_permutation=self.permutation, disable_inplace=True)
            loss, _ = autocasting_disable_decorator(config.skip_amp.loss)(self.criterion)(
                feat_dict=features, pred_dict=prediction, label_dict=labels, mode='train')
            return loss

    return ProtenixBackend()


def tensor_tree(value):
    """Remove numpy scalars/arrays so caches can be loaded with weights_only=True."""
    import numpy as np
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        # Object/string arrays keep numpy leaves in tolist(), so those leaves are converted too.
        return torch.as_tensor(value) if value.dtype.kind in 'biufc' else tensor_tree(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: tensor_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_tree(item) for item in value]
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return value
    raise ValueError(f'Unsupported cached feature type: {type(value).__name__}')


def match_interface(records, row):
    """Index of the single upstream interface record pairing this row's protein and ligand chains."""
    wanted = {(row['protein_chain_id'], 'protein'), (row['ligand_chain_id'], 'ligand')}
    matches = [i for i, record in enumerate(records)
               if str(record['pdb_id']) == row['structure_id'] and record['type'] == 'interface'
               and {(str(record['chain_1_id']), record['mol_1_type']),
                    (str(record['chain_2_id']), record['mol_2_type'])} == wanted]
    if len(matches) != 1:
        raise ValueError(f'{row["sample_id"]}: expected exactly one protein–ligand interface; got {len(matches)}')
    return matches[0]


def prepare(rows, targets, output, config, indices=None, bioassembly=None, mmcif=None, crop_size=-1,
            synthetic=False, provenance=None, overrides=None):
    """Featurize once per binding_key (repeats are hard links); structure and synthetic labels for train rows only.
    Cheap checks run before the output directory exists; every native call is seeded from its row's content."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite prepared data: {output}')
    if indices and not (bioassembly and mmcif):
        raise ValueError('--positive-indices requires --bioassembly-dir and --mmcif-dir')
    inputs = make_inputs(rows, targets)
    molecule_identity_check(rows, required=True)
    negatives = {r['sample_id'] for r in rows if synthetic and r['split'] == 'train' and synthetic_eligible(r)}
    for target_id in sorted({r['target_id'] for r in rows if r['sample_id'] in negatives}):
        apo_lookup(target_id, targets[target_id])
    positives = []
    if indices:
        from protenix.data.pipeline.dataset import BaseSingleDataset
        class PinnedInterfaceDataset(BaseSingleDataset):
            def _get_bioassembly_data(self, idx):
                _, assembly, path = super()._get_bioassembly_data(idx)
                return self._get_sample_indice(idx), assembly, path
        structural = PinnedInterfaceDataset(
            mmcif_dir=str(mmcif), bioassembly_dict_dir=str(bioassembly),
            indices_fpath=str(indices), pdb_list=[],
            cropping_configs={'crop_size': crop_size, 'method_weights': [0., 0., 1.],
                              'remove_metal': False},
            random_sample_if_failed=False, ref_pos_augment=True, lig_atom_rename=False,
            find_pocket=True, name='fragment_positive')
        records = structural.indices_list.to_dict('records')
        positives = [(row, match_interface(records, row)) for row in rows
                     if row['split'] == 'train' and structure_eligible(row)]

    import torch
    from protenix.data.inference.infer_dataloader import InferenceDataset
    from .training import seed_step

    def save(row, kind, fields):
        stamp = {'binding_key': binding_key(row)} if kind == 'binding' else {'row_hash': row_hash(row)}
        path = output / packet_name(row, kind)
        with path.open('xb') as stream:
            torch.save({'version': 1, 'kind': kind, **stamp, **tensor_tree(fields)}, stream)
        return path

    output.mkdir(parents=True)
    for row, index in positives:
        # Augmentation and crop are drawn once here; seeding by content makes reruns identical.
        seed_step(int(row_hash(row)[:16], 16))
        # process_one raises on failure, unlike loaders that silently resample.
        batch = structural.process_one(index, return_atom_token_array=True)
        if not {row['ligand_chain_id'], row['protein_chain_id']} <= set(batch['cropped_atom_array'].chain_id):
            raise ValueError(f'{row["sample_id"]}: crop removed the screening ligand or receptor')
        save(row, 'structure', {key: batch[key] for key in ('input_feature_dict', 'label_dict', 'label_full_dict')})

    input_path = output / 'inputs.json'
    write_json(input_path, inputs)
    config.input_json_path, config.dump_dir = str(input_path), str(output)
    used = [targets[target_id] for target_id in {r['target_id'] for r in rows}]
    config.use_msa = any(t.get('pairedMsaPath') or t.get('unpairedMsaPath') or t.get('msa') for t in used)
    config.use_template = any(t.get('templatesPath') for t in used)
    dataset = InferenceDataset(config)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[binding_key(row)].append(i)
    for key, members in groups.items():
        seed_step(int(key[:16], 16))
        data, atoms, error = dataset[members[0]]
        if error:
            raise ValueError(f'{rows[members[0]]["sample_id"]}: native featurization failed: {error}')
        first = save(rows[members[0]], 'binding', {'input_feature_dict': data['input_feature_dict']})
        for i in members[1:]:
            path = output / packet_name(rows[i], 'binding')
            try:
                os.link(first, path)
            except OSError:  # no hard links on this filesystem; 'xb' still refuses to overwrite
                with first.open('rb') as source, path.open('xb') as stream:
                    shutil.copyfileobj(source, stream)
        for i in members:
            if rows[i]['sample_id'] in negatives:
                save(rows[i], 'synthetic', native_synthetic_packet(
                    rows[i], targets[rows[i]['target_id']], data['input_feature_dict'], atoms))
    write_json(output / 'preparation.json', {
        'source': provenance, 'model_name': config.model_name, 'upstream_overrides': overrides or {},
        'binding_samples': len(rows), 'binding_featurizations': len(groups),
        'positive_structures': len(positives),
        'structure_msa': 'dummy; native structure dataset is prepared without MSA/templates',
        'binding_msa': bool(config.use_msa), 'binding_templates': bool(config.use_template),
        'crop_size': crop_size, 'synthetic_enabled': synthetic,
        'targets': targets, 'inputs_sha256': file_hash(input_path),
        'artifacts': {path.name: file_hash(path) for path in sorted(output.glob('*.pt'))}})
