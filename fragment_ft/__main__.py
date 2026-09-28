"""Run from the project directory: python -m fragment_ft --help."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys

from .data import (assign_splits, file_hash, make_inputs, predictable, prediction_report,
                   read_manifest, row_hash, split_audit, summary, task_name, trainable,
                   validate_rows, verify_features, write_json, write_manifest)


def parser():
    root = argparse.ArgumentParser(description='General screening data and Protenix fine-tuning (no auto-install)')
    commands = root.add_subparsers(dest='command', required=True)
    validate = commands.add_parser('validate', help='Validate labels, identities and existing split leakage; no ML dependencies')
    validate.add_argument('manifest')
    audit = commands.add_parser('audit', help='Check split leakage and report eligible/held-out observations')
    audit.add_argument('manifest'); audit.add_argument('--output')
    merge = commands.add_parser('merge', help='Merge reviewed manifests without dropping observations')
    merge.add_argument('manifests', nargs='+'); merge.add_argument('--output', required=True)
    fetch = commands.add_parser('fetch', help='Explicitly download a public resource with SHA256 receipt')
    fetch.add_argument('url'); fetch.add_argument('--output', required=True)
    fetch.add_argument('--sha256'); fetch.add_argument('--max-mb', type=int, default=512)
    pubchem = commands.add_parser('fetch-pubchem', help='Download one AID and its description; outcomes still require review')
    pubchem.add_argument('--aid', type=int, required=True); pubchem.add_argument('--output', required=True)
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
    split.add_argument('--validation-fraction', type=float, default=0.15)
    split.add_argument('--test-fraction', type=float, default=0.15)
    inputs = commands.add_parser('inputs', help='Export native inference JSON with a shared protein context per target')
    inputs.add_argument('manifest'); inputs.add_argument('--targets', required=True)
    inputs.add_argument('--output', required=True)
    compare = commands.add_parser('compare', help='Compare prediction reports on exactly the same examples')
    compare.add_argument('predictions', nargs='+'); compare.add_argument('--output', required=True)
    compare.add_argument('--top-k', type=int, default=20)

    prepare = commands.add_parser('prepare', help='Prepare binding tensors and optional positive structure tensors; needs Protenix')
    prepare.add_argument('manifest'); prepare.add_argument('--targets', required=True)
    prepare.add_argument('--output', required=True)
    prepare.add_argument('--positive-indices', help='CSV from upstream scripts/prepare_training_data.py')
    prepare.add_argument('--bioassembly-dir'); prepare.add_argument('--mmcif-dir')
    prepare.add_argument('--crop-size', type=int, default=-1)
    prepare.add_argument('--synthetic', action='store_true', help='Also prepare artificial negative labels using target apo structures')
    runtime_options(prepare)

    train = commands.add_parser('train', help='Train a frozen binding head or jointly fine-tune selected backbone modules')
    train.add_argument('manifest'); train.add_argument('--features', required=True)
    train.add_argument('--base-checkpoint', required=True)
    train.add_argument('--output', required=True)
    starting = train.add_mutually_exclusive_group()
    starting.add_argument('--resume')
    starting.add_argument('--init-checkpoint', help='Transfer learned weights; reset optimizer/step and match outputs by task')
    train.add_argument('--sampling', choices=['assay', 'uniform'], default='assay')
    train.add_argument('--mode', choices=['head', 'joint'], default='head')
    train.add_argument('--trainable-prefix', action='append', default=[])
    train.add_argument('--hidden', type=int, default=128)
    train.add_argument('--steps', type=int, default=1000)
    train.add_argument('--eval-every', type=int, default=50)
    train.add_argument('--accumulate', type=int, default=4)
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
    train.add_argument('--top-k', type=int, default=20)
    runtime_options(train)

    predict = commands.add_parser('predict', help='Predict binding scores and per-target metrics from a trained delta checkpoint')
    predict.add_argument('manifest'); predict.add_argument('--features', required=True)
    predict.add_argument('--checkpoint', required=True); predict.add_argument('--base-checkpoint', required=True)
    predict.add_argument('--protenix-source'); predict.add_argument('--output', required=True)
    predict.add_argument('--split', choices=['train', 'val', 'test', 'all'], default='test')
    predict.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    predict.add_argument('--precision', choices=['fp32', 'bf16'], default='bf16')
    predict.add_argument('--top-k', type=int, default=20)
    export = commands.add_parser('export', help='Merge trained backbone deltas into a native Protenix checkpoint for pose inference')
    export.add_argument('--checkpoint', required=True); export.add_argument('--base-checkpoint', required=True)
    export.add_argument('--protenix-source'); export.add_argument('--output', required=True)
    return root


def runtime_options(command):
    command.add_argument('--protenix-source', help='Existing Protenix source directory; never installed by this tool')
    command.add_argument('--model-name', choices=['protenix_base_default_v1.0.0', 'protenix-v2'],
                         default='protenix_base_default_v1.0.0')
    command.add_argument('--upstream-config', help='JSON overrides for upstream model/runtime configuration')


def read_json(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def read_targets(path):
    targets = read_json(path)
    root = Path(path).resolve().parent
    for target in targets.values():
        for field in ('pairedMsaPath', 'unpairedMsaPath', 'templatesPath'):
            if target.get(field):
                resolved = (root / target[field]).resolve()
                if not resolved.is_file():
                    raise ValueError(f'Missing target context file: {resolved}')
                target[field] = str(resolved)
        if target.get('apo'):
            resolved = (root / target['apo']['path']).resolve()
            if not resolved.is_file():
                raise ValueError(f'Missing apo structure: {resolved}')
            target['apo']['path'] = str(resolved)
    return targets


def require_positive_args(args):
    for name in ('steps', 'eval_every', 'accumulate', 'hidden', 'head_lr', 'backbone_lr', 'clip_grad', 'top_k'):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'--{name.replace("_", "-")} must be positive and finite')
    for name in ('structure_weight', 'weight_decay', 'negative_weight'):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'--{name.replace("_", "-")} must be nonnegative and finite')
    if args.mode == 'joint' and args.structure_weight == 0:
        raise ValueError('Joint mode requires a positive structural loss weight')
    if not math.isfinite(args.synthetic_distance) or args.synthetic_distance <= 0:
        raise ValueError('--synthetic-distance must be positive and finite')
    if args.negative_mode == 'synthetic' and (args.mode != 'joint' or args.negative_weight <= 0):
        raise ValueError('Synthetic comparison requires joint mode and positive --negative-weight')


def restore(args):
    import torch
    from .protenix_adapter import build_config, native_backend
    from .training import FineTuner, load_delta
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    metadata = payload['metadata']
    if file_hash(args.base_checkpoint) != metadata['base_checkpoint_sha256']:
        raise ValueError('Base checkpoint hash differs from the training run')
    config = build_config(metadata['model_name'], metadata['upstream_overrides'])
    model = FineTuner(native_backend(config, args.base_checkpoint), mode=metadata['mode'],
                      trainable_prefixes=metadata['trainable_prefix'], hidden=metadata['hidden'],
                      task_names=metadata.get('task_names', ['xray:hit']))
    load_delta(model, payload)
    return model, metadata


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        rows = read_manifest(args.manifest) if hasattr(args, 'manifest') else None
        if args.command == 'validate':
            print(json.dumps(summary(rows), indent=2)); return 0
        if args.command == 'audit':
            report = split_audit(rows)
            if args.output:
                write_json(args.output, report)
            print(json.dumps(report, indent=2)); return 0
        if args.command == 'merge':
            combined = [row for path in args.manifests for row in read_manifest(path)]
            validate_rows(combined)
            write_manifest(args.output, combined); return 0
        if args.command in ('fetch', 'fetch-pubchem', 'import-csv', 'import-lit'):
            from .sources import download, fetch_pubchem, import_csv, import_lit
            if args.command == 'fetch':
                result = download(args.url, args.output, args.sha256, args.max_mb * 1024 * 1024)
            elif args.command == 'fetch-pubchem':
                result = fetch_pubchem(args.aid, args.output, args.with_compounds)
            elif args.command == 'import-csv':
                result = summary(import_csv(args.input, read_json(args.mapping), args.output, args.compounds))
            else:
                result = summary(import_lit(args.active, args.inactive, read_json(args.mapping), args.compounds, args.output))
            print(json.dumps(result, indent=2)); return 0
        if args.command == 'split':
            result = assign_splits(rows, args.seed, args.validation_fraction, args.test_fraction, args.strategy)
            write_manifest(args.output, result)
            print(json.dumps(summary(result), indent=2)); return 0
        if args.command == 'inputs':
            write_json(args.output, make_inputs([r for r in rows if predictable(r)], read_targets(args.targets))); return 0
        if args.command == 'compare':
            comparison, expected = {}, None
            for path in args.predictions:
                report = read_json(path)
                records = report['predictions']
                identity = sorted(row_hash({k: v for k, v in r.items() if k not in ('probability', 'task')})
                                  for r in records)
                if expected is not None and identity != expected:
                    raise ValueError('Comparison reports contain different examples, labels or splits')
                expected = identity
                comparison[str(Path(path))] = {
                    'metrics_by_target': prediction_report(records, args.top_k),
                    'training': report.get('training', {})}
            write_json(args.output, comparison); return 0

        # Import the optional integration only for commands that actually need it.
        from .protenix_adapter import build_config, connect_source, native_backend, prepare
        provenance = connect_source(args.protenix_source)
        if args.command == 'prepare':
            validate_rows(rows, require_splits=True)
            rows = [r for r in rows if predictable(r)]
            if not rows:
                raise ValueError('No protein observations remain after excluding phenotype/double-cold cross pairs')
            if args.crop_size != -1 and args.crop_size < 1:
                raise ValueError('--crop-size must be -1 (full complex) or a positive token count')
            config = build_config(args.model_name, read_json(args.upstream_config) if args.upstream_config else {})
            prepare(rows, read_targets(args.targets), args.output, config, args.positive_indices,
                    args.bioassembly_dir, args.mmcif_dir, args.crop_size, args.synthetic, provenance)
            print(f'Prepared tensors in {args.output}'); return 0

        import torch
        from .training import FineTuner, initialize_delta, predict, run_training, save_checkpoint, seed_step
        if args.command == 'train':
            require_positive_args(args)
            validate_rows(rows, require_splits=True)
            selected = [r for r in rows if r['split'] in ('train', 'val') and trainable(r)]
            preparation = verify_features(args.features, selected, structure=args.mode == 'joint',
                                          synthetic=args.negative_mode == 'synthetic')
            if preparation['model_name'] != args.model_name:
                raise ValueError('Prepared data and training model_name differ')
            overrides = read_json(args.upstream_config) if args.upstream_config else {}
            config = build_config(args.model_name, overrides)
            seed_step(args.seed)
            tasks = sorted({task_name(r) for r in selected if r['split'] == 'train'})
            model = FineTuner(native_backend(config, args.base_checkpoint), mode=args.mode,
                              trainable_prefixes=args.trainable_prefix, hidden=args.hidden, task_names=tasks)
            base_hash = file_hash(args.base_checkpoint)
            initial_hash = None
            if args.init_checkpoint:
                initial = torch.load(args.init_checkpoint, map_location='cpu', weights_only=True)
                previous = initial['metadata']
                if previous['base_checkpoint_sha256'] != base_hash or previous['model_name'] != args.model_name or previous['upstream_overrides'] != overrides:
                    raise ValueError('Initialization must use the same base checkpoint, native model and upstream config')
                initialize_delta(model, initial)
                initial_hash = file_hash(args.init_checkpoint)
            elif args.resume:
                resumed = torch.load(args.resume, map_location='cpu', weights_only=True)
                initial_hash = resumed['metadata'].get('init_checkpoint_sha256')
            settings = ('mode', 'trainable_prefix', 'hidden', 'accumulate', 'head_lr',
                        'backbone_lr', 'weight_decay', 'structure_weight', 'clip_grad',
                        'seed', 'precision', 'eval_every', 'top_k', 'negative_mode',
                        'negative_weight', 'synthetic_distance', 'placement_seed', 'sampling')
            metadata = {key: getattr(args, key) for key in settings} | {
                'source': provenance, 'model_name': args.model_name,
                'upstream_overrides': overrides,
                'base_checkpoint_sha256': base_hash, 'task_names': tasks,
                'init_checkpoint_sha256': initial_hash,
                'manifest_sha256': file_hash(args.manifest),
                'preparation_sha256': file_hash(Path(args.features) / 'preparation.json'),
                'world_size': int(os.environ.get('WORLD_SIZE', '1'))}
            run_training(model, rows, args.features, args, metadata)
            return 0

        model, metadata = restore(args)
        if args.command == 'export':
            save_checkpoint(args.output, {'model': model.backend.model.state_dict(),
                                           'fragment_finetuning': metadata})
            print(f'Native structure weights exported to {args.output}; binding head remains in the fine-tuning checkpoint')
            return 0
        selected = [r for r in rows if predictable(r) and (args.split == 'all' or r['split'] == args.split)]
        if not selected:
            raise ValueError('No rows selected for prediction')
        preparation = verify_features(args.features, selected)
        if preparation['model_name'] != metadata['model_name']:
            raise ValueError('Prepared data and checkpoint model_name differ')
        device = torch.device(args.device)
        model.to(device)
        predictions = predict(model, selected, args.features, device, args.precision)
        write_json(args.output, {'predictions': predictions,
                                'metrics_by_target': prediction_report(predictions, args.top_k),
                                'training': metadata,
                                'checkpoint_sha256': file_hash(args.checkpoint),
                                'manifest_sha256': file_hash(args.manifest),
                                'preparation_sha256': file_hash(Path(args.features) / 'preparation.json'),
                                'score_meaning': 'Task/endpoint-specific screening score; not Kd or an assay-independent binding probability'})
        return 0
    except (ValueError, RuntimeError, OSError, ImportError, KeyError, csv.Error) as error:
        print(f'error: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
