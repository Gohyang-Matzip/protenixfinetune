"""Frozen pooled embeddings with portable provenance and verified disk storage."""
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from .data import binding_key, file_hash, packet_name, predictable, read_json, verify_features, write_json
from .training import SEED_SCHEME, autocast, encoder_rng, load_packet, pool_interface, save_checkpoint


def metadata_hash(metadata):
    return hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()


@torch.no_grad()
def extract_embeddings(backend, rows, directory, output, provenance, device,
                       precision='fp32', shard_index=0, shard_count=1):
    """Encode sorted unique inputs once; shards partition keys, not observations."""
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError('Require 0 <= shard-index < shard-count')
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite embeddings: {output}')
    selected = [r for r in rows if predictable(r)]
    if not selected:
        raise ValueError('No predictable observations to embed')
    preparation = verify_features(directory, selected)
    unique = {}
    for row in selected:
        key = binding_key(row)
        digest = preparation['artifacts'][packet_name(row, 'binding')]
        if key in unique and unique[key][1] != digest:
            raise ValueError(f'Conflicting prepared features for binding key: {key}')
        unique[key] = (row, digest)
    keys = sorted(unique)
    identity = dict(provenance, c_s=backend.c_s, c_z=backend.c_z,
                    pooling='interface_mean_v1', seed_scheme=SEED_SCHEME,
                    effective_precision='bf16' if device.type == 'cuda' and precision == 'bf16' else 'fp32')
    backend.to(device).eval().requires_grad_(False)
    output.mkdir(parents=True)
    entries = {}
    for key in keys[shard_index::shard_count]:
        row, digest = unique[key]
        features = load_packet(directory, row, 'binding', device)['input_feature_dict']
        with encoder_rng(key, device), autocast(device, precision):
            single, pair = backend.encode(features)
            pooled = pool_interface(single, pair, features).detach().cpu()
        if not torch.isfinite(pooled).all():
            raise ValueError(f'Non-finite embedding: {key}')
        path = output / (key + '.pt')
        save_checkpoint(path, pooled)
        entries[key] = {'sha256': file_hash(path), 'feature_sha256': digest}
    metadata = {'version': 1, 'identity': identity, 'entries': entries,
                'preparation_sha256': file_hash(Path(directory) / 'preparation.json'),
                'preparation': preparation, 'keys': keys,
                'shard_index': shard_index, 'shard_count': shard_count}
    write_json(output / 'embeddings.json', dict(metadata, integrity=metadata_hash(metadata)))
    return metadata


class DiskEmbeddings:
    """Validate immutable shards once, then serve pooled CPU vectors from RAM."""
    def __init__(self, paths, rows, features=None, expected=None):
        self.entries = {}
        self.vectors = {}
        self.identity = None
        self.receipts = []
        preparation_hash = None
        for directory in paths:
            directory = Path(directory)
            metadata = read_json(directory / 'embeddings.json')
            integrity = metadata.pop('integrity', None)
            if metadata.get('version') != 1 or integrity != metadata_hash(metadata):
                raise ValueError(f'Corrupt embedding metadata: {directory}')
            identity = metadata['identity']
            if self.identity is not None and identity != self.identity:
                raise ValueError('Incompatible embedding shards')
            if preparation_hash is not None and metadata['preparation_sha256'] != preparation_hash:
                raise ValueError('Incompatible embedding feature provenance')
            preparation_hash = metadata['preparation_sha256']
            if identity['pooling'] != 'interface_mean_v1' or identity['seed_scheme'] != SEED_SCHEME:
                raise ValueError('Unsupported embedding pooling or seed scheme')
            self.identity = identity
            keys = metadata['keys']
            count, index = metadata['shard_count'], metadata['shard_index']
            if count < 1 or not 0 <= index < count or keys != sorted(set(keys)) or set(metadata['entries']) != set(keys[index::count]):
                raise ValueError('Invalid embedding shard coverage')
            self.receipts.append(integrity)
            for key, entry in metadata['entries'].items():
                if len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
                    raise ValueError('Invalid embedding key')
                path = directory / (key + '.pt')
                if file_hash(path) != entry['sha256']:
                    raise ValueError(f'Corrupt embedding: {path}')
                if key in self.entries:
                    raise ValueError(f'Duplicate embedding key across shards: {key}')
                self.entries[key] = (path, entry)
                vector = torch.load(path, map_location='cpu', weights_only=True)
                width = 2 * (identity['c_s'] + identity['c_z'])
                if not isinstance(vector, torch.Tensor) or vector.shape != (width,) or vector.dtype != torch.float32 or not torch.isfinite(vector).all():
                    raise ValueError(f'Invalid pooled embedding: {path}')
                self.vectors[key] = vector
        if not self.identity:
            raise ValueError('At least one embedding shard is required')
        if expected is not None and self.identity != expected:
            raise ValueError('Embedding identity differs from checkpoint')
        missing = {binding_key(r) for r in rows} - self.entries.keys()
        if missing:
            raise ValueError(f'Missing embeddings for {len(missing)} binding keys')
        if features:
            preparation = verify_features(features, rows)
            for row in rows:
                if preparation['artifacts'][packet_name(row, 'binding')] != self.entries[binding_key(row)][1]['feature_sha256']:
                    raise ValueError('Stale embedding: prepared features changed')

    def __contains__(self, key):
        return key in self.entries

    def __getitem__(self, key):
        return self.vectors[key]


class CachedBackend(nn.Module):
    """Dimension-only backend: cached head execution cannot invoke native encoding."""
    def __init__(self, identity):
        super().__init__()
        self.c_s, self.c_z = identity['c_s'], identity['c_z']

    def encode(self, features):
        raise ValueError('Missing disk embedding; native fallback is forbidden')
