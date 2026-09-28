"""Optional adapter for upstream Protenix commit 4c355be4553512f72453ecbfb65e69f4c35d1413.

No model implementation is vendored and no installation/download is performed.
"""
import copy
import importlib.util
import os
from pathlib import Path
import sys
import subprocess

from .data import file_hash, make_inputs, row_hash, synthetic_eligible, write_json

UPSTREAM_COMMIT = '4c355be4553512f72453ecbfb65e69f4c35d1413'


def connect_source(source=None):
    if source:
        root = Path(source).resolve()
        if not (root / 'protenix/model/protenix.py').is_file() or not (root / 'configs/configs_base.py').is_file():
            raise ValueError('--protenix-source must point to an existing upstream source checkout')
        sys.path.insert(0, str(root))
    if importlib.util.find_spec('protenix') is None:
        raise RuntimeError('Protenix is not available. No installation was attempted. '
                           'Use this command later in a configured training environment '
                           'with --protenix-source /path/to/Protenix.')
    # Avoid custom extension compilation; CUDA triangle kernels remain configurable.
    os.environ.setdefault('LAYERNORM_TYPE', 'torch')
    spec = importlib.util.find_spec('protenix')
    actual_root = Path(spec.origin).resolve().parent.parent
    commit = subprocess.run(['git', '-C', str(actual_root), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True, check=False)
    return {'expected_upstream_commit': UPSTREAM_COMMIT,
            'actual_source': str(actual_root),
            'actual_commit': commit.stdout.strip() if commit.returncode == 0 else None}


def deep_update(destination, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(destination.get(key), dict):
            deep_update(destination[key], value)
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
    deep_update(base, {'model_name': model_name, 'train_confidence_only': False,
                      'triangle_attention': 'torch', 'triangle_multiplicative': 'torch',
                      'enable_diffusion_shared_vars_cache': False,
                      'enable_efficient_fusion': False, 'diffusion_batch_size': 4,
                      'model': {'N_cycle': 4}, 'sample_diffusion': {'N_sample': 1},
                      'data': {'template': {'fetch_remote': False}}})
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
        return torch.as_tensor(value) if value.dtype.kind not in 'OUS' else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: tensor_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_tree(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'Unsupported cached feature type: {type(value).__name__}')


def prepare(rows, targets, output, config, indices=None, bioassembly=None, mmcif=None, crop_size=-1,
            synthetic=False, provenance=None):
    import torch
    from protenix.data.inference.infer_dataloader import InferenceDataset

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    input_path = output / 'inputs.json'
    write_json(input_path, make_inputs(rows, targets))
    config.input_json_path, config.dump_dir = str(input_path), str(output)
    config.use_msa = any(t.get('pairedMsaPath') or t.get('unpairedMsaPath') or t.get('msa')
                         for t in targets.values())
    config.use_template = any(t.get('templatesPath') for t in targets.values())
    dataset = InferenceDataset(config)
    for i, row in enumerate(rows):
        data, atoms, error = dataset[i]
        if error:
            raise ValueError(f'{row["sample_id"]}: native featurization failed: {error}')
        packet = {'version': 1, 'kind': 'binding', 'row_hash': row_hash(row),
                  'input_feature_dict': tensor_tree(data['input_feature_dict'])}
        with (output / f'{row["sample_id"]}.binding.pt').open('xb') as stream:
            torch.save(packet, stream)
        if synthetic and synthetic_eligible(row):
            from .synthetic import native_synthetic_packet
            artificial = native_synthetic_packet(row, targets[row['target_id']], data['input_feature_dict'], atoms)
            with (output / f'{row["sample_id"]}.synthetic.pt').open('xb') as stream:
                torch.save(artificial, stream)

    prepared_structures = 0
    if indices:
        if not bioassembly or not mmcif:
            raise ValueError('Structural preparation requires --bioassembly-dir and --mmcif-dir')
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
        for row in rows:
            if row['label'] != '1' or not row.get('structure_id'):
                continue
            table = structural.indices_list
            matches = []
            for i, record in enumerate(table.to_dict('records')):
                if str(record['pdb_id']) != row['structure_id'] or record['type'] != 'interface':
                    continue
                chains = {(str(record['chain_1_id']), record['mol_1_type']),
                          (str(record['chain_2_id']), record['mol_2_type'])}
                if chains == {(row['protein_chain_id'], 'protein'), (row['ligand_chain_id'], 'ligand')}:
                    matches.append(i)
            if len(matches) != 1:
                raise ValueError(f'{row["sample_id"]}: expected exactly one protein–ligand interface; got {len(matches)}')
            # process_one raises on failure, unlike loaders that silently resample.
            batch = structural.process_one(matches[0], return_atom_token_array=True)
            atoms = batch['cropped_atom_array']
            if not {row['ligand_chain_id'], row['protein_chain_id']} <= set(atoms.chain_id):
                raise ValueError(f'{row["sample_id"]}: crop removed the screening ligand or receptor')
            packet = {'version': 1, 'kind': 'structure', 'row_hash': row_hash(row),
                      **{key: tensor_tree(batch[key]) for key in
                         ('input_feature_dict', 'label_dict', 'label_full_dict')}}
            with (output / f'{row["sample_id"]}.structure.pt').open('xb') as stream:
                torch.save(packet, stream)
            prepared_structures += 1
    write_json(output / 'preparation.json', {
        'source': provenance, 'model_name': config.model_name,
        'binding_samples': len(rows), 'positive_structures': prepared_structures,
        'structure_msa': 'dummy; native structure dataset is prepared without MSA/templates',
        'binding_msa': bool(config.use_msa), 'binding_templates': bool(config.use_template),
        'crop_size': crop_size, 'synthetic_enabled': synthetic,
        'targets': targets, 'inputs_sha256': file_hash(input_path),
        'artifacts': {path.name: file_hash(path) for path in sorted(output.glob('*.pt'))}})
