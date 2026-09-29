"""Run from the project directory: python -m fragment_ft --help."""
import argparse
from collections import Counter
import csv
import http.client
import json
import math
import os
from pathlib import Path
import sys

from .data import (assign_splits, cross_audit, file_hash, make_inputs, molecule_identity_check, predictable,
                   prediction_report, read_json, read_manifest, row_hash, split_audit, summary,
                   task_name, training_rows, validate_rows, verify_features, write_json, write_manifest)


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f'must be a positive integer, got {text}')
    return value


def parser():
    root = argparse.ArgumentParser(description='General screening data and Protenix fine-tuning (no auto-install)')
    commands = root.add_subparsers(dest='command', required=True)
    from .baselines import register_parser
    register_parser(commands)
    validate = commands.add_parser('validate', help='Validate labels, identities and existing split leakage; no ML dependencies')
    validate.add_argument('manifest')
    audit = commands.add_parser('audit', help='Check split leakage and report eligible/held-out observations')
    audit.add_argument('manifest'); audit.add_argument('--output')
    audit.add_argument('--against', help='Reference manifest (e.g. pretraining): report val/test rows here that overlap its train rows')
    merge = commands.add_parser('merge', help='Merge reviewed manifests without dropping observations')
    merge.add_argument('manifests', nargs='+'); merge.add_argument('--output', required=True)
    fetch = commands.add_parser('fetch', help='Explicitly download a public resource with SHA256 receipt')
    fetch.add_argument('url'); fetch.add_argument('--output', required=True)
    fetch.add_argument('--sha256'); fetch.add_argument('--max-mb', type=positive_int, default=512)
    pubchem = commands.add_parser('fetch-pubchem', help='Download one AID and its description; outcomes still require review')
    pubchem.add_argument('--aid', type=positive_int, required=True); pubchem.add_argument('--output', required=True)
    pubchem.add_argument('--with-compounds', action='store_true')
    importing = commands.add_parser('import-csv', help='Apply an explicit outcome/column mapping to public observations')
    importing.add_argument('input'); importing.add_argument('--mapping', required=True)
    importing.add_argument('--compounds'); importing.add_argument('--output', required=True)
    lit = commands.add_parser('import-lit', help='Import LIT-PCBA active/inactive SMILES with reviewed metadata')
    lit.add_argument('--active', required=True); lit.add_argument('--inactive', required=True)
    lit.add_argument('--mapping', required=True); lit.add_argument('--compounds', required=True)
    lit.add_argument('--output', required=True)
    split = commands.add_parser('split', help='Assign chemistry groups consistently across all targets')
    split.add_argument('manifest'); split.add_argument('--output', required=True)
    split.add_argument('--seed', type=int, default=42)
    split.add_argument('--strategy', choices=['chemistry', 'target', 'both'], default='chemistry')
    split.add_argument('--validation-fraction', type=float, default=0.15,
                       help='Cap on the share of trainable rows, filled with whole groups (at least one)')
    split.add_argument('--test-fraction', type=float, default=0.15,
                       help='Cap on the share of trainable rows, filled with whole groups (at least one)')
    inputs = commands.add_parser('inputs', help='Export native inference JSON with a shared protein context per target')
    inputs.add_argument('manifest'); inputs.add_argument('--targets', required=True)
    inputs.add_argument('--output', required=True)
    compare = commands.add_parser('compare', help='Compare prediction reports on exactly the same examples')
    compare.add_argument('predictions', nargs='+'); compare.add_argument('--output', required=True)
    compare.add_argument('--top-k', type=positive_int, default=20)

    prepare = commands.add_parser('prepare', help='Prepare binding tensors and optional positive structure tensors; needs Protenix')
    prepare.add_argument('manifest'); prepare.add_argument('--targets', required=True)
    prepare.add_argument('--output', required=True)
    prepare.add_argument('--positive-indices', help='CSV from upstream scripts/prepare_training_data.py')
    prepare.add_argument('--bioassembly-dir'); prepare.add_argument('--mmcif-dir')
    prepare.add_argument('--crop-size', type=int, default=-1)
    prepare.add_argument('--synthetic', action='store_true', help='Also prepare artificial negative labels using target apo structures')
    runtime_options(prepare)

    train = commands.add_parser('train', help='Train a frozen binding head or jointly fine-tune selected backbone modules')
    train.add_argument('manifest'); train.add_argument('--features')
    train.add_argument('--base-checkpoint')
    train.add_argument('--embeddings', nargs='+', help='Verified embedding shard directories; head mode only')
    train.add_argument('--output', required=True)
    starting = train.add_mutually_exclusive_group()
    starting.add_argument('--resume')
    starting.add_argument('--init-checkpoint', help='Transfer learned weights; reset optimizer/step and match outputs by task')
    train.add_argument('--sampling', choices=['assay', 'uniform'], default='assay')
    train.add_argument('--mode', choices=['head', 'joint'], default='head')
    train.add_argument('--trainable-prefix', action='append', default=[])
    train.add_argument('--hidden', type=positive_int, default=128)
    train.add_argument('--steps', type=positive_int, default=1000)
    train.add_argument('--eval-every', type=positive_int, default=50)
    train.add_argument('--accumulate', type=positive_int, default=4)
    train.add_argument('--head-lr', type=float, default=1e-4)
    train.add_argument('--backbone-lr', type=float, default=1e-6)
    train.add_argument('--weight-decay', type=float, default=0.01)
    train.add_argument('--structure-weight', type=float, default=1.0)
    train.add_argument('--negative-mode', choices=['classifier', 'synthetic'], default='classifier')
    train.add_argument('--negative-weight', type=float, default=1.0)
    train.add_argument('--synthetic-distance', type=float, default=40., help='Minimum receptor–ligand bounding-sphere clearance, Angstrom')
    train.add_argument('--placement-seed', type=int, default=10000)
    train.add_argument('--clip-grad', type=float, default=1.0)
    train.add_argument('--seed', type=int, default=42)
    train.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    train.add_argument('--precision', choices=['fp32', 'bf16'], default='bf16')
    train.add_argument('--top-k', type=positive_int, default=20)
    train.add_argument('--select-metric', choices=['average_precision', 'log_loss'], default='average_precision',
                       help='Validation score marked best_so_far: macro AP (higher) or macro log loss (lower)')
    train.add_argument('--dist-timeout-minutes', type=positive_int, default=60,
                       help='Collective timeout for multi-GPU runs; validation is sharded across ranks')
    runtime_options(train)

    predict = commands.add_parser('predict', help='Predict binding scores and per-target metrics from a trained delta checkpoint')
    predict.add_argument('manifest'); predict.add_argument('--features')
    predict.add_argument('--embeddings', nargs='+', help='Verified embedding shard directories')
    predict.add_argument('--checkpoint', required=True); predict.add_argument('--base-checkpoint')
    predict.add_argument('--protenix-source'); predict.add_argument('--output', required=True)
    predict.add_argument('--split', choices=['train', 'val', 'test', 'all'], default='test')
    predict.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    predict.add_argument('--precision', choices=['fp32', 'bf16'], default='bf16')
    predict.add_argument('--top-k', type=positive_int, default=20)
    predict.add_argument('--allow-different-manifest', action='store_true',
                         help='Predict a manifest other than the training one; its val/test rows may have been trained on')
    embed = commands.add_parser('embed', help='Extract frozen pooled CPU embeddings once per binding key')
    embed.add_argument('manifest'); embed.add_argument('--features', required=True)
    embed.add_argument('--base-checkpoint', required=True); embed.add_argument('--output', required=True)
    embed.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    embed.add_argument('--precision', choices=['fp32', 'bf16'], default='fp32')
    embed.add_argument('--shard-index', type=int, default=0)
    embed.add_argument('--shard-count', type=positive_int, default=1)
    runtime_options(embed)
    export = commands.add_parser('export', help='Merge trained backbone deltas into a native Protenix checkpoint for pose inference')
    export.add_argument('--checkpoint', required=True); export.add_argument('--base-checkpoint', required=True)
    export.add_argument('--protenix-source'); export.add_argument('--output', required=True)
    return root


def runtime_options(command):
    command.add_argument('--protenix-source', help='Existing Protenix source directory; never installed by this tool')
    command.add_argument('--model-name', choices=['protenix_base_default_v1.0.0', 'protenix-v2'],
                         default='protenix_base_default_v1.0.0')
    command.add_argument('--upstream-config', help='JSON overrides for upstream model/runtime configuration')


def read_targets(path):
    targets = read_json(path)
    if not isinstance(targets, dict) or not all(isinstance(t, dict) for t in targets.values()):
        raise ValueError(f'{path}: targets JSON must map each target_id to an object')
    root = Path(path).resolve().parent
    def resolve(value, exists, what):  # paths are relative to the targets file, not the working directory
        if not isinstance(value, str):
            raise ValueError(f'{target_id}: {what} path must be a string')
        resolved = (root / value).resolve()
        if not exists(resolved):
            raise ValueError(f'Missing {what}: {resolved}')
        return str(resolved)
    for target_id, target in targets.items():
        if not isinstance(target.get('sequence', ''), str):
            raise ValueError(f'{target_id}: sequence must be a string')
        for field in ('pairedMsaPath', 'unpairedMsaPath', 'templatesPath'):
            if target.get(field):
                target[field] = resolve(target[field], Path.is_file, 'target context file')
        msa = target.get('msa')  # legacy upstream key
        if msa is not None and not isinstance(msa, dict):
            raise ValueError(f'{target_id}: msa must be an object')
        if msa and msa.get('precomputed_msa_dir'):
            msa['precomputed_msa_dir'] = resolve(msa['precomputed_msa_dir'], Path.is_dir, 'target MSA directory')
        apo = target.get('apo')
        if apo:
            if not isinstance(apo, dict) or not apo.get('path'):
                raise ValueError(f'{target_id}: apo must be an object with a path')
            apo['path'] = resolve(apo['path'], Path.is_file, 'apo structure')
    return targets


def check_train_args(args):
    """Cheap flag checks, before features are hashed or the base checkpoint is loaded."""
    if getattr(args, 'embeddings', None) and (args.mode != 'head' or args.negative_mode != 'classifier'):
        raise ValueError('Disk embeddings require head/classifier mode')
    for name in ('head_lr', 'backbone_lr', 'clip_grad', 'synthetic_distance'):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'--{name.replace("_", "-")} must be positive and finite')
    for name in ('structure_weight', 'weight_decay', 'negative_weight'):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'--{name.replace("_", "-")} must be nonnegative and finite')
    if (args.mode == 'head') == bool(args.trainable_prefix):
        raise ValueError('head mode forbids --trainable-prefix; joint mode requires it')
    if args.mode == 'joint' and args.structure_weight == 0:
        raise ValueError('Joint mode requires a positive structural loss weight')
    if args.negative_mode == 'synthetic' and (args.mode != 'joint' or args.negative_weight <= 0):
        raise ValueError('Synthetic comparison requires joint mode and positive --negative-weight')
    if not args.resume and Path(args.output).exists():
        raise FileExistsError(f'Refusing to overwrite run directory: {args.output}')


def warn_commit(what, source, provenance):
    if (source or {}).get('actual_commit') != provenance.get('actual_commit'):
        print(f'warning: {what} used Protenix commit {(source or {}).get("actual_commit")}, '
              f'current is {provenance.get("actual_commit")}', file=sys.stderr)


def check_preparation(preparation, model_name, overrides, provenance):
    """Features are fixed at prepare time: model and data.* overrides must match the consumer."""
    if preparation['model_name'] != model_name:
        raise ValueError('Prepared data and model_name differ')
    if 'upstream_overrides' not in preparation:  # caches from before prepare recorded them
        print('warning: Prepared data does not record its upstream overrides; data.* settings are not compared',
              file=sys.stderr)
    elif (preparation['upstream_overrides'] or {}).get('data') != (overrides or {}).get('data'):
        raise ValueError('Prepared data used different data.* upstream overrides; use the same --upstream-config for prepare and train')
    warn_commit('Prepared data', preparation.get('source'), provenance)


def restore(args, payload):
    from .protenix_adapter import build_config, native_backend
    from .training import FineTuner, load_delta
    metadata = payload['metadata']
    if file_hash(args.base_checkpoint) != metadata['base_checkpoint_sha256']:
        raise ValueError('Base checkpoint hash differs from the training run')
    config = build_config(metadata['model_name'], metadata['upstream_overrides'])
    model = FineTuner(native_backend(config, args.base_checkpoint), mode=metadata['mode'],
                      trainable_prefixes=metadata['trainable_prefix'], hidden=metadata['hidden'],
                      task_names=metadata.get('task_names', ['xray:hit']))
    load_delta(model, payload)
    return model


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == 'baseline':
            from .baselines import run
            return run(args)
        rows = read_manifest(args.manifest) if hasattr(args, 'manifest') else None
        if args.command == 'validate':
            print(json.dumps(summary(rows), indent=2)); return 0
        if args.command == 'audit':
            report = split_audit(rows)
            report['molecule_identity'] = molecule_identity_check(rows)  # checked=False without RDKit
            if args.against:
                report['cross_corpus'] = cross_audit(rows, read_manifest(args.against))
            if args.output:
                write_json(args.output, report)
            print(json.dumps(report, indent=2)); return 0
        if args.command == 'merge':
            combined = [row for path in args.manifests for row in read_manifest(path)]
            if len({bool(r.get('split')) for r in combined}) > 1:
                raise ValueError('Merge either all-split or all-unsplit manifests; split after merging')
            validate_rows(combined)
            write_manifest(args.output, combined)
            print(json.dumps({'inputs_sha256': {path: file_hash(path) for path in args.manifests},
                              **summary(combined)}, indent=2)); return 0
        if args.command in ('fetch', 'fetch-pubchem', 'import-csv', 'import-lit'):
            from .sources import download, fetch_pubchem, import_csv, import_lit
            if args.command == 'fetch':
                result = download(args.url, args.output, args.sha256, args.max_mb * 1024 * 1024)
            elif args.command == 'fetch-pubchem':
                result = fetch_pubchem(args.aid, args.output, args.with_compounds)
            elif args.command == 'import-csv':
                result = summary(import_csv(args.input, read_json(args.mapping), args.output, args.compounds,
                                            mapping_sha256=file_hash(args.mapping)))
            else:
                result = summary(import_lit(args.active, args.inactive, read_json(args.mapping), args.compounds,
                                            args.output, mapping_sha256=file_hash(args.mapping)))
            print(json.dumps(result, indent=2)); return 0
        if args.command == 'split':
            result = assign_splits(rows, args.seed, args.validation_fraction, args.test_fraction, args.strategy)
            write_manifest(args.output, result)
            print(json.dumps(summary(result), indent=2)); return 0
        if args.command == 'inputs':
            write_json(args.output, make_inputs([r for r in rows if predictable(r)], read_targets(args.targets))); return 0
        if args.command == 'compare':
            comparison, expected, trained_on = {}, None, set()
            for path in args.predictions:
                report = read_json(path)
                records = report.get('predictions') if isinstance(report, dict) else None
                if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
                    raise ValueError(f'{path}: expected a predict report with a "predictions" list')
                examples = [{k: v for k, v in r.items() if k not in ('probability', 'task')} for r in records]
                if any(not isinstance(v, str) for r in examples for v in r.values()):
                    raise ValueError(f'{path}: prediction fields other than probability must be strings, as predict writes them')
                validate_rows(examples)
                identity = sorted(row_hash(r) for r in examples)
                if expected is not None and identity != expected:
                    raise ValueError('Comparison reports contain different examples, labels or splits')
                expected = identity
                scored = prediction_report(records, args.top_k)
                if not scored:
                    raise ValueError(f'{path}: no reliable 0/1 observations to score')
                training = report.get('training') or {}
                if not isinstance(training, dict):
                    raise ValueError(f'{path}: "training" must be an object')
                trained_on.add(training.get('manifest_sha256'))
                same = report.get('same_manifest_as_training')
                if same is False:
                    print(f'warning: {path} evaluated a manifest other than its training manifest; '
                          'its held-out rows may have been trained on', file=sys.stderr)
                comparison[str(Path(path))] = {'metrics_by_target': scored, 'training': training,
                                               'same_manifest_as_training': same}
            if len(trained_on - {None}) > 1:
                print('warning: compared checkpoints were trained on different manifests', file=sys.stderr)
            write_json(args.output, comparison); return 0

        # Import the optional integration only for commands that actually need it.
        cached = bool(getattr(args, 'embeddings', None))
        if args.command in ('train', 'predict') and not cached and (not args.features or not args.base_checkpoint):
            raise ValueError('Native train/predict requires --features and --base-checkpoint')
        if not cached:
            from .protenix_adapter import build_config, connect_source, native_backend, prepare
            provenance = connect_source(args.protenix_source)
        overrides = read_json(args.upstream_config) if getattr(args, 'upstream_config', None) else {}
        if not isinstance(overrides, dict):
            raise ValueError('--upstream-config must be a JSON object')
        if args.command == 'prepare':
            # adapter.prepare checks the output, targets and apo structures before it creates anything.
            validate_rows(rows, require_splits=True)
            rows = [r for r in rows if predictable(r)]
            if not rows:
                raise ValueError('No protein observations remain after excluding phenotype/double-cold cross pairs')
            if args.crop_size != -1 and args.crop_size < 1:
                raise ValueError('--crop-size must be -1 (full complex) or a positive token count')
            for path in (args.positive_indices, args.bioassembly_dir, args.mmcif_dir):
                if path and not Path(path).exists():
                    raise FileNotFoundError(f'Missing structural input: {path}')
            targets = read_targets(args.targets)
            config = build_config(args.model_name, overrides)
            prepare(rows, targets, args.output, config, args.positive_indices, args.bioassembly_dir,
                    args.mmcif_dir, args.crop_size, args.synthetic, provenance, overrides=overrides)
            print(f'Prepared tensors in {args.output}'); return 0

        import torch
        from .training import FineTuner, initialize_delta, predict, run_training, save_checkpoint, seed_step
        if args.command == 'embed':
            from .embeddings import extract_embeddings
            validate_rows(rows, require_splits=True)
            if not 0 <= args.shard_index < args.shard_count:
                raise ValueError('Require 0 <= shard-index < shard-count')
            if Path(args.output).exists():
                raise FileExistsError(f'Refusing to overwrite embeddings: {args.output}')
            preparation = verify_features(args.features, [r for r in rows if predictable(r)])
            check_preparation(preparation, args.model_name, overrides, provenance)
            identity = {'source': provenance, 'model_name': args.model_name, 'upstream_overrides': overrides,
                        'base_checkpoint_sha256': file_hash(args.base_checkpoint)}
            backend = native_backend(build_config(args.model_name, overrides), args.base_checkpoint)
            extract_embeddings(backend, rows, args.features, args.output, identity, torch.device(args.device),
                               args.precision, args.shard_index, args.shard_count)
            print(f'Embeddings written to {args.output}'); return 0
        if args.command == 'train':
            check_train_args(args)
            validate_rows(rows, require_splits=True)
            train, validation = training_rows(rows)
            embeddings = None
            if cached:
                from .embeddings import CachedBackend, DiskEmbeddings
                embeddings = DiskEmbeddings(args.embeddings, train + validation, args.features)
                identity = embeddings.identity
                if args.upstream_config and overrides != identity['upstream_overrides']:
                    raise ValueError('Embedding upstream configuration differs')
                if args.model_name != identity['model_name']:
                    raise ValueError('Embedding model_name differs')
                provenance, overrides = identity['source'], identity['upstream_overrides']
                base_hash = identity['base_checkpoint_sha256']
                if args.base_checkpoint and file_hash(args.base_checkpoint) != base_hash:
                    raise ValueError('Embedding base checkpoint differs')
            else:
                preparation = verify_features(args.features, train + validation, structure=args.mode == 'joint',
                                              synthetic=args.negative_mode == 'synthetic')
                check_preparation(preparation, args.model_name, overrides, provenance)
                base_hash = file_hash(args.base_checkpoint)
            initial = initial_hash = None
            if args.init_checkpoint:
                initial = torch.load(args.init_checkpoint, map_location='cpu', weights_only=True)
                previous = initial['metadata']
                if previous['base_checkpoint_sha256'] != base_hash or previous['model_name'] != args.model_name or previous['upstream_overrides'] != overrides:
                    raise ValueError('Initialization must use the same base checkpoint, native model and upstream config')
                if cached and previous.get('embedding_identity', identity) != identity:
                    raise ValueError('Initialization embedding identity differs')
                initial_hash = file_hash(args.init_checkpoint)
            elif args.resume:
                resumed = torch.load(args.resume, map_location='cpu', weights_only=True)
                initial_hash = resumed['metadata'].get('init_checkpoint_sha256')
            tasks = sorted({task_name(r) for r in train})
            settings = ('mode', 'trainable_prefix', 'hidden', 'accumulate', 'head_lr',
                        'backbone_lr', 'weight_decay', 'structure_weight', 'clip_grad',
                        'seed', 'precision', 'eval_every', 'top_k', 'negative_mode',
                        'negative_weight', 'synthetic_distance', 'placement_seed', 'sampling', 'select_metric')
            metadata = {key: getattr(args, key) for key in settings} | {
                'source': provenance, 'model_name': args.model_name,
                'upstream_overrides': overrides,
                'base_checkpoint_sha256': base_hash, 'task_names': tasks,
                'init_checkpoint_sha256': initial_hash,
                'manifest_sha256': file_hash(args.manifest),
                'preparation_sha256': file_hash(Path(args.features) / 'preparation.json') if not cached else None,
                'world_size': int(os.environ.get('WORLD_SIZE', '1')), 'seed_scheme': 2}
            seed_step(args.seed)
            if cached:
                metadata.update(embedding_identity=embeddings.identity, embedding_shards=sorted(embeddings.receipts),
                                embedding_features={key: entry['feature_sha256']
                                                    for key, (_, entry) in embeddings.entries.items()})
                backend = CachedBackend(embeddings.identity)
            else:
                backend = native_backend(build_config(args.model_name, overrides), args.base_checkpoint)
            model = FineTuner(backend, mode=args.mode, trainable_prefixes=args.trainable_prefix,
                              hidden=args.hidden, task_names=tasks, embeddings=embeddings)
            if initial:
                initialize_delta(model, initial)
            run_training(model, rows, args.features, args, metadata)
            return 0

        # predict/export: every check runs before the base checkpoint is loaded or inference starts.
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        metadata = payload['metadata']
        if not cached:
            warn_commit('Checkpoint', metadata.get('source'), provenance)
        if Path(args.output).exists():
            raise FileExistsError(f'Refusing to overwrite: {args.output}')
        if args.command == 'export':
            model = restore(args, payload)
            save_checkpoint(args.output, {'model': model.backend.model.state_dict(),
                                           'fragment_finetuning': metadata})
            print(f'Native structure weights exported to {args.output}; binding head remains in the fine-tuning checkpoint')
            return 0
        same_manifest = file_hash(args.manifest) == metadata.get('manifest_sha256')
        if not same_manifest and not args.allow_different_manifest:
            raise ValueError('Manifest differs from the training manifest, so its held-out rows may have been trained on; '
                             'pass --allow-different-manifest to predict anyway')
        trained = set(metadata.get('task_names', ['xray:hit']))
        selected = [r for r in rows if predictable(r) and (args.split == 'all' or r['split'] == args.split)]
        untrained = dict(Counter(task_name(r) for r in selected if task_name(r) not in trained))
        selected = [r for r in selected if task_name(r) in trained]
        if not selected:
            raise ValueError(f'No rows of trained tasks selected for prediction; untrained: {untrained}')
        if untrained:
            print(f'warning: skipping rows of tasks this checkpoint never trained: {untrained}', file=sys.stderr)
        if cached:
            from .embeddings import CachedBackend, DiskEmbeddings
            from .training import load_delta
            if metadata['mode'] != 'head':
                raise ValueError('Disk embeddings cannot predict a joint checkpoint')
            embeddings = DiskEmbeddings(args.embeddings, selected, args.features, metadata.get('embedding_identity'))
            for key, digest in metadata.get('embedding_features', {}).items():
                if key in embeddings and embeddings.entries[key][1]['feature_sha256'] != digest:
                    raise ValueError('Stale embedding: feature provenance differs from checkpoint')
            for key in ('base_checkpoint_sha256', 'model_name', 'upstream_overrides'):
                if embeddings.identity[key] != metadata[key]:
                    raise ValueError(f'Embedding {key} differs from checkpoint')
            if args.base_checkpoint and file_hash(args.base_checkpoint) != metadata['base_checkpoint_sha256']:
                raise ValueError('Embedding base checkpoint differs')
            model = FineTuner(CachedBackend(embeddings.identity), hidden=metadata['hidden'],
                              task_names=metadata.get('task_names', ['xray:hit']), embeddings=embeddings)
            load_delta(model, payload)
        else:
            preparation = verify_features(args.features, selected)
            check_preparation(preparation, metadata['model_name'], metadata.get('upstream_overrides'), provenance)
            model = restore(args, payload)
        device = torch.device(args.device)
        model.to(device)
        predictions = predict(model, selected, args.features, device, args.precision)
        write_json(args.output, {'predictions': predictions,
                                'metrics_by_target': prediction_report(predictions, args.top_k),
                                'untrained_tasks': untrained,
                                'same_manifest_as_training': same_manifest,
                                'training': metadata,
                                'checkpoint_sha256': file_hash(args.checkpoint),
                                'manifest_sha256': file_hash(args.manifest),
                                'preparation_sha256': file_hash(Path(args.features) / 'preparation.json') if not cached else None,
                                'embedding_shards': sorted(embeddings.receipts) if cached else None,
                                'score_meaning': 'Task/endpoint-specific screening score; not Kd or an assay-independent binding probability'})
        return 0
    except (ValueError, RuntimeError, OSError, ImportError, KeyError, csv.Error, http.client.HTTPException) as error:
        print(f'error: missing key {error}' if isinstance(error, KeyError) else f'error: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
