"""Portable command wrapper around existing strict-isolation BRIDGE stages."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from configs import load_config

def json_read(path):
    return json.loads(Path(path).read_text())

def validate_split_contract(cfg, split):
    """Fail on a declaration mismatch; never regenerate or relabel saved IDs."""
    names = ('warmup', 'validation', 'sft')
    ids = {name: list(map(str, json_read(split / (name + '_ids.json')))) for name in names}
    sets = {name: set(values) for name, values in ids.items()}
    if any(not values or len(values) != len(sets[name]) for name, values in ids.items()):
        raise ValueError('Split IDs must be nonempty and unique')
    if any(sets[names[i]] & sets[names[j]] for i in range(3) for j in range(i+1, 3)):
        raise ValueError('Split IDs must be disjoint')
    counts = {name: len(values) for name, values in ids.items()}
    if cfg.expected_split_sizes is not None and counts != cfg.expected_split_sizes:
        raise ValueError(f'Manuscript split declaration {cfg.expected_split_sizes} differs '
                         f'from saved split {counts}. No resplitting is permitted here. '
                         'Use the explicit artifact profile for historical reproduction; '
                         'paper reproduction requires verified matching split provenance.')
    return counts

def build_plan(args):
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    gpu_ids = [s.strip() for s in args.gpus.split(',')]
    if (not gpu_ids or any(not s.isdigit() for s in gpu_ids)
            or len(set(gpu_ids)) != len(gpu_ids)):
        raise ValueError('--gpus requires unique explicit nonnegative integer IDs')
    if cfg.fsdp and len(gpu_ids) != cfg.num_gpus:
        raise ValueError(f'This FSDP profile requires {cfg.num_gpus} GPUs, got {len(gpu_ids)}')
    split = Path(args.split_dir or cfg.data_split_dir).resolve()
    out = Path(args.output_dir).resolve()
    warmup = str(Path(args.reuse_warmup).resolve()) if args.reuse_warmup else str(out / 'warmup/final_model')
    direction = str(Path(args.reuse_direction).resolve()) if args.reuse_direction else str(out / 'l2/direction/v_l2_ret_lambda_0p2.pt')
    scoring_cfg = out / 'configs/scoring.yaml'
    target_cfg = out / 'configs/target_direction.yaml'
    final_cfg = out / 'configs/final.yaml'
    stage_driver = ROOT / 'scripts/reproduce.py'
    commands = []
    def stage(name, config, dest, **options):
        command = [sys.executable, str(stage_driver), 'stage', name, '--config', str(config),
                   '--split-dir', str(split), '--output-dir', str(dest)]
        for key, value in options.items():
            command += ['--' + key.replace('_', '-'), str(value)]
        if cfg.fsdp and name in ('warmup', 'train'):
            command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                       '--nproc_per_node', str(len(gpu_ids)), *command[1:]]
        commands.append((name, command))
    if not args.reuse_warmup:
        stage('warmup', scoring_cfg, out / 'warmup')
    if not args.reuse_direction:
        stage('direction', target_cfg, out / 'target_direction', warmup_checkpoint=warmup)
        commands.append(('l2_mix', [sys.executable, str(stage_driver),
            'l2-direction', '--config', str(scoring_cfg), '--output-root', str(out / 'l2'),
            '--pretrained-checkpoint', cfg.model_name_or_path, '--warmup-checkpoint', warmup,
            '--target-direction', str(out / 'target_direction/validation_direction.pt')]))
    stage('scoring', scoring_cfg, out / 'scoring', warmup_checkpoint=warmup,
          direction_path=direction, num_gpus=len(gpu_ids))
    stage('select', scoring_cfg, out / 'selection', scores=out / 'scoring/scoring_results.pt')
    stage('train', final_cfg, out / 'train', method='bridge_closure75',
          selection=out / 'selection/selection_bridge_closure75.pt')
    if cfg.dataset_name == 'metamath':
        stage('eval', final_cfg, out / 'eval', checkpoint=out / 'train/final_model',
              math_test_dir=Path(os.environ.get('BRIDGE_DATA_ROOT', str(ROOT / 'data'))) / 'math')
    return cfg, gpu_ids, split, out, commands

def preflight(cfg, gpu_ids, split, out, args):
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite an existing run directory: {out}')
    validate_split_contract(cfg, split)
    if not Path(cfg.model_name_or_path).is_dir():
        raise FileNotFoundError('Set BRIDGE_MODEL_PATH to an original local pretrained checkpoint')
    manifest = json_read(split / 'split_manifest.json')
    declared_dataset = manifest.get('split_config', {}).get('dataset')
    if declared_dataset and declared_dataset != cfg.dataset_name:
        raise ValueError(f'Split dataset={declared_dataset}, active dataset={cfg.dataset_name}')
    if cfg.dataset_name == 'metamath':
        expected = manifest['split_config'].get('dataset_file_sha256')
        digest = hashlib.sha256()
        with Path(cfg.dataset_path).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                digest.update(chunk)
        if expected and digest.hexdigest() != expected:
            raise ValueError('Exact MetaMathQA40K file SHA256 differs from saved split provenance')
    for path in (args.reuse_warmup, args.reuse_direction):
        if path and not Path(path).exists():
            raise FileNotFoundError(path)
    usage = subprocess.check_output(['nvidia-smi', '-i', ','.join(gpu_ids),
        '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    for line in usage.splitlines():
        memory, utilization = [int(s.strip()) for s in line.split(',')]
        if memory > 256 or utilization > 5:
            raise RuntimeError(f'An assigned GPU is busy ({memory}MiB,{utilization}%); not starting')

def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=
        'Individual stages: reproduce.py stage <stage> --help; '
        'L2 mixing: reproduce.py l2-direction --help')
    parser.add_argument('--config', required=True)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--split-dir')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--scoring-seed', type=int, default=42)
    parser.add_argument('--reuse-warmup')
    parser.add_argument('--reuse-direction')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.run == args.dry_run:
        parser.error('Choose exactly one of --dry-run or --run')
    cfg, gpu_ids, split, out, commands = build_plan(args)
    plan = dict(config=args.config, final_training_seed=cfg.seed, scoring_seed=args.scoring_seed,
                expected_split_sizes=cfg.expected_split_sizes, budget_rounding=cfg.budget_rounding,
                gpus=gpu_ids, split_dir=str(split), output_dir=str(out),
                commands=[dict(stage=name, argv=command) for name, command in commands])
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return
    preflight(cfg, gpu_ids, split, out, args)
    import yaml
    (out / 'configs').mkdir(parents=True, exist_ok=False)
    final = asdict(cfg)
    final.update(output_dir=str(out), data_split_dir=str(split))
    scoring = dict(final, seed=args.scoring_seed)
    target = dict(scoring, retention_enabled=False, retention_lambda=0.0)
    for name, value in [('final', final), ('scoring', scoring), ('target_direction', target)]:
        (out / 'configs' / (name + '.yaml')).write_text(yaml.safe_dump(value, sort_keys=False))
    (out / 'command_manifest.json').write_text(json.dumps(plan, indent=2) + '\n')
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=','.join(gpu_ids), TOKENIZERS_PARALLELISM='false',
                       HF_HUB_DISABLE_TELEMETRY='1')
    (out / 'logs').mkdir()
    for name, command in commands:
        print(name, shlex.join(command), flush=True)
        with (out / 'logs' / (name + '.log')).open('x') as stream:
            subprocess.run(command, cwd=ROOT, env=environment, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
    status = 'COMPLETE' if cfg.dataset_name == 'metamath' else 'TRAINING_COMPLETE_TASK_EVAL_PENDING'
    (out / 'pipeline_status.json').write_text(json.dumps({'status': status}) + '\n')

def dispatch():
    if len(sys.argv) > 1 and sys.argv[1] == 'stage':
        sys.argv.pop(1)
        from pipeline.stages import main as stage_main
        stage_main()
    elif len(sys.argv) > 1 and sys.argv[1] == 'l2-direction':
        from pipeline.direction import main as direction_main
        direction_main()
    else:
        main()


if __name__ == '__main__':
    dispatch()
