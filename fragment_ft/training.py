"""Small binding head and PyTorch training loop. No Protenix import at module level."""
from contextlib import contextmanager, nullcontext
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random

import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .data import (binding_key, predictable, prediction_report, row_hash, synthetic_eligible,
                   task_name, trainable, validate_rows, write_json)


@contextmanager
def encoder_rng(key, device):
    """Native MSA sampling remains random in eval; isolate a stable input seed."""
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    python_state = random.getstate()
    try:
        import numpy as np
    except ImportError:
        np = None
    numpy_state = np.random.get_state() if np is not None else None
    devices = [device.index if device.index is not None else torch.cuda.current_device()] \
        if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        try:
            random.seed(seed)
            torch.default_generator.manual_seed(seed)
            for index in devices:
                torch.cuda.default_generators[index].manual_seed(seed)
            if np is not None:
                np.random.seed(seed)
            yield
        finally:
            random.setstate(python_state)
            if np is not None:
                np.random.set_state(numpy_state)


def pool_interface(single, pair, features):
    if single.ndim != 2 or pair.ndim != 3:
        raise ValueError('Expected one unbatched complex: single[N,C], pair[N,N,C]')
    mapping = features['atom_to_token_idx'].long()
    protein = torch.unique(mapping[features['is_protein'].bool()])
    ligand = torch.unique(mapping[features['is_ligand'].bool()])
    if not len(protein) or not len(ligand):
        raise ValueError('Both protein and ligand tokens are required')
    # Token-weighted pooling avoids overweighting residues with more atoms.
    return torch.cat((single[protein].mean(0), single[ligand].mean(0),
                      pair[protein[:, None], ligand[None, :]].mean((0, 1)),
                      pair[ligand[:, None], protein[None, :]].mean((0, 1)))).float()


class FineTuner(nn.Module):
    def __init__(self, backend, mode='head', trainable_prefixes=(), hidden=128, task_names=('xray:hit',)):
        super().__init__()
        if mode not in ('head', 'joint') or hidden < 1:
            raise ValueError('mode must be head or joint and hidden must be positive')
        if mode == 'head' and trainable_prefixes:
            raise ValueError('head mode cannot unfreeze backbone parameters')
        if mode == 'joint' and not trainable_prefixes:
            raise ValueError('joint mode requires explicit --trainable-prefix values')
        self.backend, self.mode = backend, mode
        self.task_names = tuple(task_names)
        if not self.task_names or len(set(self.task_names)) != len(self.task_names):
            raise ValueError('task_names must be nonempty and unique')
        self._pooled_cache = {}
        self._inherited_parameter_names = set()
        self.backend.requires_grad_(False)
        for prefix in trainable_prefixes:
            matched = [parameter for name, parameter in backend.model.named_parameters()
                       if name == prefix or name.startswith(prefix + '.')]
            if not matched:
                raise ValueError(f'No backbone parameter matches prefix {prefix!r}')
            for parameter in matched:
                parameter.requires_grad_(True)
        width = 2 * backend.c_s + 2 * backend.c_z
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden),
                                  nn.GELU(), nn.Linear(hidden, len(self.task_names)))
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        if self.mode == 'head':
            self.backend.eval()
        return self

    def forward(self, features, positive_structure=None, step=0, cache_key=None,
                synthetic_structure=None, synthetic_weight=1.0, task='xray:hit'):
        if task not in self.task_names:
            raise ValueError(f'Unknown/untrained task: {task}')
        if self.mode == 'head' and (positive_structure is not None or synthetic_structure is not None):
            raise ValueError('head mode does not perform structure training')
        if self.mode == 'head' and cache_key in self._pooled_cache:
            pooled = self._pooled_cache[cache_key].to(next(self.head.parameters()).device)
        else:
            stable_encoding = cache_key is not None and (self.mode == 'head' or not self.training)
            context = encoder_rng(cache_key, next(self.head.parameters()).device) if stable_encoding else nullcontext()
            with context, torch.set_grad_enabled(self.training and self.mode == 'joint'):
                single, pair = self.backend.encode(features)
                pooled = pool_interface(single, pair, features)
            if self.mode == 'head' and cache_key is not None:
                self._pooled_cache[cache_key] = pooled.detach().cpu()
        logit = self.head(pooled)[self.task_names.index(task)]
        structural = logit.new_zeros(())
        if positive_structure is not None:
            structural = self.backend.structure_loss(positive_structure, step)
        if synthetic_structure is not None:
            structural = structural + synthetic_weight * self.backend.structure_loss(synthetic_structure, step)
        return logit, structural


def training_loss(logit, label, structural, structure_weight=1.0, sample_weight=1.0):
    if label not in (0, 1) or not math.isfinite(structure_weight) or structure_weight < 0:
        raise ValueError('Only reliable binary labels and nonnegative finite weights are trainable')
    if not math.isfinite(sample_weight) or not 0 <= sample_weight <= 1:
        raise ValueError('sample_weight must be between 0 and 1')
    binding = nn.functional.binary_cross_entropy_with_logits(logit.float(), logit.new_tensor(float(label)).float())
    return sample_weight * binding + structure_weight * structural.float()


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [to_device(item, device) for item in value]
    return value


def load_packet(directory, row, kind, device):
    if kind not in ('binding', 'structure', 'synthetic'):
        raise ValueError('Unknown prepared packet kind')
    if kind == 'structure' and (row['label'] != '1' or not row.get('structure_id')):
        raise ValueError('Only positive experimental complexes can enter the structure branch')
    if kind == 'synthetic' and not synthetic_eligible(row):
        raise ValueError('Only reliable X-ray negatives enter the synthetic comparison branch')
    path = Path(directory) / f'{row["sample_id"]}.{kind}.pt'
    packet = torch.load(path, map_location='cpu', weights_only=True)
    if packet.get('version') != 1 or packet.get('kind') != kind or packet.get('row_hash') != row_hash(row):
        raise ValueError(f'Stale/mismatched prepared packet: {path}')
    return to_device(packet, device)


def checkpoint_payload(model, optimizer, step, metadata, best):
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    names |= model._inherited_parameter_names
    delta = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items() if name in names}
    return {'version': 1, 'delta': delta, 'optimizer': optimizer.state_dict(),
            'inherited_parameter_names': sorted(model._inherited_parameter_names),
            'step': step, 'metadata': metadata, 'best_validation_loss': best}


def load_delta(model, payload):
    if payload.get('version') != 1:
        raise ValueError('Unsupported fine-tuning checkpoint version')
    if tuple(payload['metadata'].get('task_names', ['xray:hit'])) != model.task_names:
        raise ValueError('Checkpoint task ordering differs from this model')
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    inherited = set(payload.get('inherited_parameter_names', []))
    if not inherited <= {name for name, _ in model.named_parameters()}:
        raise ValueError('Checkpoint inherited parameters differ from this model')
    expected |= inherited
    if set(payload['delta']) != expected:
        raise ValueError('Checkpoint trainable parameter set differs from this model')
    model.load_state_dict(payload['delta'], strict=False)
    model._inherited_parameter_names = inherited


@torch.no_grad()
def initialize_delta(model, payload):
    """Transfer learned weights, matching output rows by task; reset optimizer/step."""
    if payload.get('version') != 1:
        raise ValueError('Unsupported fine-tuning checkpoint version')
    source_tasks = payload['metadata'].get('task_names', ['xray:hit'])
    state = model.state_dict()
    final = f'head.{len(model.head)-1}.'
    shared = set(source_tasks) & set(model.task_names)
    for name, tensor in payload['delta'].items():
        if name not in state:
            raise ValueError(f'Initialization checkpoint has an unknown parameter: {name}')
        if name in (final+'weight', final+'bias'):
            if tensor.shape[0] != len(source_tasks) or tensor.shape[1:] != state[name].shape[1:]:
                raise ValueError('Initialization head shape differs; preserve --hidden')
            for task in shared:
                state[name][model.task_names.index(task)].copy_(tensor[source_tasks.index(task)])
        else:
            if state[name].shape != tensor.shape:
                raise ValueError(f'Initialization parameter shape differs: {name}')
            state[name].copy_(tensor)
            if name.startswith('backend.'):
                model._inherited_parameter_names.add(name)


def save_checkpoint(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite checkpoint: {path}')
    partial = path.with_suffix('.partial')
    with partial.open('xb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    partial.rename(path)


def autocast(device, precision):
    return torch.autocast(device_type='cuda', dtype=torch.bfloat16) \
        if precision == 'bf16' and device.type == 'cuda' else nullcontext()


@torch.no_grad()
def predict(model, rows, feature_dir, device, precision='fp32'):
    model.eval()
    predictions = []
    for row in rows:
        if not predictable(row):
            raise ValueError('Phenotypic/excluded observations are not protein predictions')
        features = load_packet(feature_dir, row, 'binding', device)['input_feature_dict']
        with autocast(device, precision):
            logit, _ = model(features, cache_key=binding_key(row), task=task_name(row))
        probability = logit.float().sigmoid().item()
        if not math.isfinite(probability):
            raise ValueError(f'{row["sample_id"]}: non-finite prediction')
        predictions.append(dict(row, probability=probability, task=task_name(row)))
    return predictions


def seed_step(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Native Protenix uses numpy as well. CPU-only unit tests do not need it.
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed % (2**32))


def run_training(model, rows, feature_dir, args, metadata):
    validate_rows(rows, require_splits=True)
    train = [r for r in rows if r['split'] == 'train' and trainable(r)]
    validation = [r for r in rows if r['split'] == 'val' and trainable(r)]
    positives = [r for r in train if r['label'] == '1' and r.get('structure_id')]
    negatives = [r for r in train if synthetic_eligible(r)]
    if {r['label'] for r in train} != {'0', '1'} or {r['label'] for r in validation} != {'0', '1'}:
        raise ValueError('Training and validation each need both reliable classes; revise group splits or collect more hits')
    if args.mode == 'joint' and not positives:
        raise ValueError('Joint training needs positive experimental training structures')
    training_tasks = {task_name(r) for r in train}
    if training_tasks != set(model.task_names) or {task_name(r) for r in validation} != training_tasks:
        raise ValueError('Training, validation and model must have the same task set')
    for task in training_tasks:
        for name, records in (('train', train), ('val', validation)):
            if {r['label'] for r in records if task_name(r) == task} != {'0', '1'}:
                raise ValueError(f'{task}: {name} needs both reliable classes')
    if args.negative_mode == 'synthetic' and not negatives:
        raise ValueError('Synthetic comparison needs reliable X-ray training negatives')
    buckets = {}
    for row in train:
        target = buckets.setdefault(task_name(row), {}).setdefault(row['target_id'], {})
        target.setdefault((row.get('source', ''), row.get('assay_id', '')), []).append(row)
    # Select task, then target, then assay uniformly; large campaigns cannot dominate by row count.
    buckets = [[list(target.values()) for target in task.values()] for task in buckets.values()]
    world, rank = int(os.environ.get('WORLD_SIZE', '1')), int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device = torch.device(f'cuda:{local_rank}' if args.device == 'cuda' else args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    output = Path(args.output)
    try:
        if rank == 0:
            output.mkdir(parents=True, exist_ok=bool(args.resume))
        if world > 1:
            dist.barrier()
        model.to(device)
        head_parameters = list(model.head.parameters())
        backbone_parameters = [p for p in model.backend.parameters() if p.requires_grad]
        groups = [{'params': head_parameters, 'lr': args.head_lr}]
        if backbone_parameters:
            groups.append({'params': backbone_parameters, 'lr': args.backbone_lr})
        optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
        start, best = 0, float('inf')
        if args.resume:
            payload = torch.load(args.resume, map_location='cpu', weights_only=True)
            if payload['metadata'] != metadata:
                raise ValueError('Resume metadata differs: preserve data, base weights, model and training settings')
            load_delta(model, payload)
            optimizer.load_state_dict(payload['optimizer'])
            start, best = payload['step'], payload['best_validation_loss']
        if start >= args.steps:
            raise ValueError('--steps must exceed the checkpoint step')
        if rank == 0:
            write_json(output / f'run_from_{start:06d}.json', metadata)
        wrapped = DistributedDataParallel(model, device_ids=[local_rank] if device.type == 'cuda' else None,
                                           find_unused_parameters=True) if world > 1 else model
        for step in range(start, args.steps):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running = 0.0
            for micro in range(args.accumulate):
                seed = args.seed + (step * args.accumulate + micro) * world + rank
                seed_step(seed)
                rng = random.Random(seed)
                row = rng.choice(rng.choice(rng.choice(rng.choice(buckets)))) \
                    if args.sampling == 'assay' else rng.choice(train)
                features = load_packet(feature_dir, row, 'binding', device)['input_feature_dict']
                structure = load_packet(feature_dir, rng.choice(positives), 'structure', device) \
                    if args.mode == 'joint' else None
                synthetic = None
                if args.negative_mode == 'synthetic':
                    from .synthetic import place_packet
                    synthetic = place_packet(load_packet(feature_dir, rng.choice(negatives), 'synthetic', device),
                                             args.synthetic_distance, seed + args.placement_seed)
                sync = wrapped.no_sync() if world > 1 and micro + 1 < args.accumulate else nullcontext()
                with sync:
                    with autocast(device, args.precision):
                        logit, structural = wrapped(features, structure, step, cache_key=binding_key(row),
                                                    synthetic_structure=synthetic,
                                                    synthetic_weight=args.negative_weight, task=task_name(row))
                        loss = training_loss(logit, int(row['label']), structural, args.structure_weight,
                                             float(row.get('weight') or 1))
                    if not torch.isfinite(loss):
                        raise ValueError(f'Non-finite loss at step {step}, sample {row["sample_id"]}')
                    (loss / args.accumulate).backward()
                running += loss.detach().float().item() / args.accumulate
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                          args.clip_grad, error_if_nonfinite=True)
            optimizer.step()
            if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
                if rank == 0:
                    predictions = predict(model, validation, feature_dir, device, args.precision)
                    report = prediction_report(predictions, k=args.top_k)
                    # Equal task weight, then equal weight for each target/assay within task.
                    task_losses = defaultdict(list)
                    for item in report.values():
                        task_losses[item['task']].append(item['log_loss'])
                    score = sum(sum(values)/len(values) for values in task_losses.values()) / len(task_losses)
                    improved = score < best
                    best = min(best, score)
                    checkpoint = output / f'step_{step + 1:06d}.pt'
                    save_checkpoint(checkpoint, checkpoint_payload(model, optimizer, step + 1, metadata, best))
                    write_json(output / f'step_{step + 1:06d}.json', {
                        'step': step + 1, 'rank0_train_loss': running, 'validation': report,
                        'macro_validation_log_loss': score, 'best_so_far': improved,
                        'checkpoint': checkpoint.name, 'predictions': predictions})
                    print(json.dumps({'step': step + 1, 'validation_loss': score,
                                      'best_so_far': improved, 'checkpoint': str(checkpoint)}), flush=True)
                if world > 1:
                    dist.barrier()
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()
