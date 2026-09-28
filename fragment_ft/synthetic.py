"""Opt-in artificial unbound-state ablation, never experimental ground truth."""
import copy
from functools import lru_cache
import math

from .data import synthetic_eligible


def residue_mapping(sequence_ids, apo):
    def integer(value):
        if isinstance(value, bool) or not (isinstance(value, int) or
                                          isinstance(value, str) and value.lstrip('-').isdigit()):
            raise ValueError('Apo residue identifiers and offset must be integers')
        return int(value)

    ids = {integer(value) for value in sequence_ids}
    offset = integer(apo.get('residue_offset', 0))
    overrides = {integer(key): integer(value) for key, value in apo.get('residue_map', {}).items()}
    if set(overrides) - ids:
        raise ValueError('Apo residue_map contains unknown sequence residue identifiers')
    mapping = {key: overrides.get(key, key + offset) for key in ids}
    if len(set(mapping.values())) != len(mapping):
        raise ValueError('Apo residue mapping must be one-to-one')
    return mapping


def detached_coordinates(coordinates, protein_mask, ligand_mask, clearance=40.0, seed=42):
    """Keep receptor fixed; rotate ligand rigidly and put bounding spheres apart."""
    import torch
    if not math.isfinite(clearance) or clearance <= 0:
        raise ValueError('Synthetic clearance must be positive and finite')
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or not torch.isfinite(coordinates).all():
        raise ValueError('Coordinates must be finite [N_atom,3]')
    protein_mask, ligand_mask = protein_mask.bool(), ligand_mask.bool()
    if not protein_mask.any() or not ligand_mask.any() or (protein_mask & ligand_mask).any():
        raise ValueError('Nonempty disjoint observed protein and ligand masks required')
    receptor, ligand = coordinates[protein_mask], coordinates[ligand_mask]
    center = receptor.mean(0)
    centered = ligand - ligand.mean(0)
    receptor_radius = torch.linalg.vector_norm(receptor - center, dim=-1).max()
    ligand_radius = torch.linalg.vector_norm(centered, dim=-1).max()
    generator = torch.Generator(device='cpu').manual_seed(seed)
    quaternion = torch.randn(4, generator=generator, dtype=torch.float64)
    w, x, y, z = (quaternion / quaternion.norm()).tolist()
    rotation = coordinates.new_tensor([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    direction = torch.randn(3, generator=generator).to(coordinates)
    direction = direction / direction.norm()
    result = coordinates.clone()
    result[ligand_mask] = centered @ rotation.T + center + direction * (receptor_radius + ligand_radius + clearance)
    return result


def place_packet(packet, clearance, seed):
    """Only label coordinates change. Native input conformers/templates stay identical."""
    if packet.get('kind') != 'synthetic':
        raise ValueError('Only explicitly synthetic packets can be relocated')
    # Copy only the label dicts that change; feature tensors are shared read-only, not duplicated.
    result = dict(packet, label_dict=dict(packet['label_dict']), label_full_dict=dict(packet['label_full_dict']))
    features = result['input_feature_dict']
    labels = result['label_dict']
    coordinates = detached_coordinates(
        labels['coordinate'], features['is_protein'].bool() & labels['coordinate_mask'].bool(),
        features['is_ligand'].bool(), clearance, seed)
    if result['label_full_dict']['coordinate'].shape != coordinates.shape:
        raise ValueError('Synthetic ablation requires an uncropped full-complex label')
    labels['coordinate'] = coordinates
    result['label_full_dict']['coordinate'] = coordinates.clone()
    return result


# ponytail: 16 parsed apo chains are kept; add grouping by target if corpora exceed that.
@lru_cache(maxsize=16)
def apo_atoms(path, chain_id):
    """Observed apo atoms by (res_id, atom_name), parsed once per structure."""
    from biotite.structure.io import load_structure
    observed = load_structure(path, model=1, altloc='occupancy')
    observed = observed[observed.chain_id == chain_id]
    if not len(observed):
        raise ValueError(f'No apo atoms in chain {chain_id}')
    if any(str(code).strip() for code in observed.ins_code):
        raise ValueError('Apo insertion codes need explicit renumbering before synthetic preparation')
    lookup = {}
    for atom in observed:
        key = (int(atom.res_id), str(atom.atom_name))
        if key in lookup:
            raise ValueError(f'Ambiguous apo atom {key}; resolve alternate conformations first')
        lookup[key] = atom
    return lookup


def apo_lookup(target_id, target):
    """Checked apo settings and atom lookup; prepare calls this for every target before featurizing."""
    apo = target.get('apo')
    if not apo or not apo.get('path') or not apo.get('chain_id'):
        raise ValueError(f'{target_id}: synthetic preparation requires apo.path and apo.chain_id')
    if target.get('count', 1) != 1 or target.get('modifications'):
        raise ValueError('Synthetic preparation currently requires one unmodified protein chain')
    return apo, apo_atoms(apo['path'], apo['chain_id'])


def native_synthetic_packet(row, target, features, atom_array):
    """Map observed apo atoms onto the native sequence/SMILES atom order.

    Supports one noncovalent ligand and one unmodified protein chain. Explicit
    residue mapping avoids guessing sequence alignments or insertion codes.
    """
    import numpy as np
    import torch
    from protenix.data.core.featurizer import Featurizer
    from protenix.data.utils import data_type_transform, make_dummy_feature

    if not synthetic_eligible(row):
        raise ValueError('Synthetic labels require a reliable X-ray negative')
    apo, lookup = apo_lookup(row['target_id'], target)
    array = atom_array.copy()
    protein = features['is_protein'].bool()
    ligand = features['is_ligand'].bool()
    if torch.unique(features['asym_id'][torch.unique(features['atom_to_token_idx'][ligand])]).numel() != 1:
        raise ValueError('Synthetic preparation requires one ligand chain')
    coordinates = np.zeros_like(array.coord)
    mask = np.zeros(len(array), dtype=np.int64)
    mapping = residue_mapping([int(atom.res_id) for i, atom in enumerate(array) if protein[i]], apo)
    ca_total = ca_matched = 0
    for i, atom in enumerate(array):
        if not protein[i]:
            continue
        ca_total += atom.atom_name == 'CA'
        index = mapping[int(atom.res_id)]
        observed_atom = lookup.get((index, str(atom.atom_name)))
        if observed_atom is None:
            continue
        if observed_atom.res_name != atom.res_name or observed_atom.element != atom.element:
            raise ValueError(f'{row["sample_id"]}: apo residue/element mismatch at sequence residue {atom.res_id}')
        coordinates[i] = observed_atom.coord
        mask[i] = 1
        ca_matched += atom.atom_name == 'CA'
    coverage = ca_matched / max(1, ca_total)
    minimum = float(apo.get('minimum_ca_fraction', 0.8))
    if not 0 < minimum <= 1 or coverage < minimum:
        raise ValueError(f'{row["sample_id"]}: apo CA coverage {coverage:.1%}; check sequence and residue mapping')
    if not features['ref_mask'][ligand].bool().all():
        raise ValueError('Ligand reference conformer contains unresolved atoms')
    coordinates[ligand.numpy()] = features['ref_pos'][ligand].numpy()
    mask[ligand.numpy()] = 1
    array.coord = coordinates
    array.set_annotation('is_resolved', mask)
    full_labels, _ = Featurizer.get_gt_full_complex_features(array)
    inputs = copy.deepcopy(features)
    # Match the positive native structure route's dummy MSA/template policy.
    for key in list(inputs):
        if key.startswith('template_'):
            del inputs[key]
    inputs = make_dummy_feature(inputs, dummy_feats=['msa', 'template'])
    inputs = data_type_transform(inputs)
    inputs['is_distillation'] = torch.tensor([False])
    # ponytail: identity atom permutations for artificial labels; add chemical
    # symmetry permutations only if ablation results justify the extra mapping.
    counts = {}
    permutations = []
    for uid in inputs['ref_space_uid'].tolist():
        permutations.append([counts.get(uid, 0)])
        counts[uid] = counts.get(uid, 0) + 1
    inputs['atom_perm_list'] = permutations
    # prepare adds the packet header and converts numpy leaves (tensor_tree).
    return {'apo_ca_coverage': coverage, 'input_feature_dict': inputs,
            'label_dict': {'coordinate': torch.tensor(coordinates), 'coordinate_mask': torch.tensor(mask)},
            'label_full_dict': full_labels}
