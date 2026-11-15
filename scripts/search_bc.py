"""Search BC settings on frozen data, replicate finalists, then open test sets.

Each worker uses one CPU thread by default. Source data/checkpoints are read-only.
Rerunning the same command resumes completed runs and optimizer checkpoints.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    from recap.generation import write_json
    write_json(path, value)


def ranking(row):
    return row['validation_success_rate'], -row['validation_mse']


def completed_stage(path, names, seeds):
    if not path.exists():
        return None
    rows = read(path)
    expected = {(name, seed) for name in names for seed in seeds}
    actual = {(row['candidate'], row['seed']) for row in rows}
    return rows if len(rows) == len(expected) and actual == expected else None


def fingerprint(source):
    paths = [source / 'config.json', source / 'scenes.json']
    for split in ('train', 'validation'):
        paths += sorted((source / 'data' / split / 'expert').glob('*.npz'))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def prepare(source, destination, settings, seed, budget, interval, device):
    cfg = copy.deepcopy(read(source / 'config.json'))
    cfg['seed'], cfg['device'] = seed, device
    cfg['train'].update(settings, max_steps=budget, validation_steps=interval, checkpoint_steps=interval)
    if destination.exists():
        previous = read(destination / 'config.json')
        previous['train']['max_steps'] = budget
        if cfg != previous:
            raise ValueError(f'Existing run has different settings: {destination}')
    else:
        destination.mkdir(parents=True)
        shutil.copy2(source / 'scenes.json', destination / 'scenes.json')
        for split in ('train', 'validation'):
            shutil.copytree(source / 'data' / split / 'expert', destination / 'data' / split / 'expert')
    write(destination / 'config.json', cfg)


def worker(root, threads):
    import random
    import numpy as np
    import torch
    from recap.cli import fit

    root = Path(root)
    cfg = read(root / 'config.json')
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    last = root / 'bc.last.pt'
    if not last.exists() or torch.load(last, map_location='cpu', weights_only=False)['global_step'] < cfg['train']['max_steps']:
        fit(root, 'bc', cfg, resume=last.exists())
    best = torch.load(root / 'bc.pt', map_location='cpu', weights_only=False)
    metrics = best['history'][-1]
    row = dict(run=str(root), seed=cfg['seed'], train=cfg['train'], best_step=best['global_step'],
               validation_success_rate=best['best_success'], validation_mse=best['best'],
               validation_by_difficulty=metrics['validation_by_difficulty'])
    write(root / 'search_result.json', row)


def launch(root, threads):
    started = time.monotonic()
    with (root / f'train-{read(root / "config.json")["train"]["max_steps"]}.log').open('a', encoding='utf-8') as log:
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', str(root), '--threads', str(threads)]
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    if result.returncode:
        raise RuntimeError(f'Training failed; see {root}/train-*.log')
    row = read(root / 'search_result.json')
    row['wall_seconds'] = time.monotonic() - started
    return row


def batch(jobs, output, workers, threads, stage):
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(launch, root, threads): name for name, root in jobs}
        for future in as_completed(pending):
            row = future.result()
            row['candidate'] = pending[future]
            results.append(row)
            write(output / f'{stage}.json', sorted(results, key=lambda r: (r['candidate'], r['seed'])))
            print('RESULT', stage, row['candidate'], 'seed', row['seed'], 'step', row['best_step'],
                  'validation', round(row['validation_success_rate'], 4), flush=True)
    return results


def export_summary(output, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import mujoco
    import numpy as np
    import torch
    import yaml

    selected = report['selected']
    rows = sorted(report['screen'], key=ranking, reverse=True)
    def compact(group):
        result = {key: group[key] for key in ('episodes', 'success_count', 'success_rate', 'collision_rate', 'timeout_rate', 'success_ci95')}
        result['by_difficulty'] = {name: {key: value[key] for key in ('episodes', 'success_count', 'success_rate', 'collision_rate', 'timeout_rate')}
                                   for name, value in group['by_difficulty'].items()}
        for key in ('completion_seconds', 'near_goal_stall_count'):
            if key in group:
                result[key] = group[key]
        return result
    summary = dict(source=report['plan']['source'], data_fingerprint=report['plan']['fingerprint'],
                   environment=read(Path(report['plan']['source']) / 'config.json')['env'],
                   versions=dict(torch=torch.__version__, mujoco=mujoco.__version__),
                   screen=[{k: r[k] for k in ('candidate', 'best_step', 'validation_success_rate', 'validation_mse', 'train')} for r in rows],
                   selected=selected['candidate'],
                   replication=[dict(candidate=r['candidate'], mean_validation_success=r['validation_success_rate'],
                                     seeds=[dict(seed=s['seed'], step=s['best_step'], validation=s['validation_success_rate']) for s in r['runs']])
                                for r in report['aggregates']], tests=[])
    for trial in report['tests']:
        groups = trial['metrics']
        total = sum(group['episodes'] for group in groups.values())
        successful = sum(group['success_count'] for group in groups.values())
        summary['tests'].append(dict(seed=trial['seed'], best_step=trial['best_step'], episodes=total,
                                     successes=successful, success_rate=successful / total,
                                     splits={name: compact(group) for name, group in groups.items()}))
    rates = [row['success_rate'] for row in summary['tests']]
    summary['mean_test_success'] = float(np.mean(rates))
    summary['std_test_success'] = float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.
    if report.get('baseline_test'):
        summary['baseline_test'] = dict(seed=report['baseline_test']['seed'],
                                       splits={name: compact(group) for name, group in report['baseline_test']['metrics'].items()})
    reference = next(row for row in selected['runs'] if row['seed'] == report['plan']['seeds'][0])
    cfg = copy.deepcopy(read(Path(reference['run']) / 'config.json'))
    cfg['device'] = 'auto'
    # The first predeclared seed is the reproducible reference recipe. Its best
    # update is already a scheduled validation point in the longer search.
    cfg['train']['max_steps'] = reference['best_step']
    summary['recommended_seed'] = reference['seed']
    summary['recommended_steps'] = reference['best_step']
    summary['recommended_train'] = cfg['train']
    with (output / 'best_config.yaml').open('w', encoding='utf-8') as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False, allow_unicode=True)
    write(output / 'summary.json', summary)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for row in report['screen']:
        history = read(Path(row['run']) / 'bc.history.json')
        history = [h for h in history if h['step'] <= report['plan']['steps']]
        axes[0].plot([h['step'] for h in history], [h['validation_success_rate'] for h in history], label=row['candidate'])
    axes[0].set(xlabel='Optimizer updates', ylabel='Validation success', ylim=(0, 1))
    axes[0].legend(fontsize=7, ncol=2)
    for group in report['aggregates']:
        for row in group['runs']:
            history = read(Path(row['run']) / 'bc.history.json')
            axes[1].plot([h['step'] for h in history], [h['validation_success_rate'] for h in history],
                         label=f'{group["candidate"]}, seed {row["seed"]}', alpha=.8)
    axes[1].set(xlabel='Optimizer updates', ylabel='Validation success', ylim=(0, 1))
    axes[1].legend(fontsize=7, ncol=2)
    for axis in axes:
        axis.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output / 'learning_curves.png', dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run')
    parser.add_argument('--output')
    parser.add_argument('--plan', help='JSON mapping candidate names to train overrides')
    parser.add_argument('--steps', type=int, default=8000)
    parser.add_argument('--extend-steps', type=int, default=20000)
    parser.add_argument('--interval', type=int, default=500)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--finalists', type=int, default=2)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.threads)
        return
    if not all((args.run, args.output, args.plan)):
        parser.error('--run, --output, and --plan are required')
    if min(args.steps, args.interval, args.workers, args.threads, args.finalists) <= 0 or args.extend_steps < args.steps:
        parser.error('Budgets/counts must be positive and extend-steps >= steps')
    if len(set(args.seeds)) != len(args.seeds) or not args.seeds:
        parser.error('Provide distinct training seeds')
    if args.device != 'cpu' and args.workers != 1:
        parser.error('Use --workers 1 for GPU training')
    source, output = Path(args.run).resolve(), Path(args.output).resolve()
    if source == output or source in output.parents or output in source.parents:
        parser.error('Source and output must be separate directories')
    candidates = read(args.plan)
    if not candidates or any(not name or Path(name).name != name or name in ('.', '..') for name in candidates):
        parser.error('Use nonempty, safe candidate directory names')
    output.mkdir(parents=True, exist_ok=True)
    plan = dict(source=str(source), fingerprint=fingerprint(source), candidates=candidates,
                steps=args.steps, extend_steps=args.extend_steps, interval=args.interval,
                seeds=args.seeds, finalists=args.finalists, device=args.device,
                selection='validation success, then MSE; replicated mean success, then mean MSE')
    manifest = output / 'search_plan.json'
    if manifest.exists() and read(manifest) != plan:
        raise ValueError('Source data or search plan changed; use a new output directory')
    write(manifest, plan)
    def jobs(names, seeds, budget):
        result = []
        for name in names:
            for seed in seeds:
                root = output / f'{name}-s{seed}'
                prepare(source, root, candidates[name], seed, budget, args.interval, args.device)
                result.append((name, root))
        return result
    # Never replace an already completed screen with later, extended checkpoints
    # when resuming the orchestration after replication has begun.
    screened = completed_stage(output / 'screen.json', candidates, args.seeds[:1])
    if screened is None:
        screened = batch(jobs(candidates, args.seeds[:1], args.steps), output, args.workers, args.threads, 'screen')
    finalists = [row['candidate'] for row in sorted(screened, key=ranking, reverse=True)[:args.finalists]]
    write(output / 'finalists.json', finalists)
    replicated = completed_stage(output / 'replication.json', finalists, args.seeds)
    if replicated is None:
        replicated = batch(jobs(finalists, args.seeds, args.extend_steps), output, args.workers, args.threads, 'replication')
    aggregates = []
    for name in finalists:
        rows = [r for r in replicated if r['candidate'] == name]
        aggregates.append(dict(candidate=name, validation_success_rate=sum(r['validation_success_rate'] for r in rows) / len(rows),
                               validation_mse=sum(r['validation_mse'] for r in rows) / len(rows), runs=rows))
    selected = max(aggregates, key=ranking)
    # Freeze selection before any test rollout. The other candidate is never tested.
    write(output / 'selection.json', dict(aggregates=aggregates, selected=selected))
    import torch
    from recap.cli import evaluate_policies
    from recap.datasets import TEST_SPLITS
    torch.set_num_threads(args.threads)
    baseline_test = None
    if 'baseline' in candidates:
        root = output / f'baseline-s{args.seeds[0]}'
        path = root / 'search_test.json'
        if path.exists():
            baseline_metrics = read(path)
        else:
            baseline_metrics = evaluate_policies(root, read(root / 'config.json'), models=['bc'], splits=list(TEST_SPLITS), checkpoint=root / 'bc.pt')['bc']
            write(path, baseline_metrics)
        baseline_test = dict(seed=args.seeds[0], metrics=baseline_metrics)
    tests = []
    for row in selected['runs']:
        root = Path(row['run'])
        report_path = root / 'search_test.json'
        if report_path.exists():
            report = read(report_path)
        else:
            report = evaluate_policies(root, read(root / 'config.json'), models=['bc'], splits=list(TEST_SPLITS), checkpoint=root / 'bc.pt')['bc']
            write(report_path, report)
        tests.append(dict(seed=row['seed'], best_step=row['best_step'], metrics=report))
    if fingerprint(source) != plan['fingerprint']:
        raise RuntimeError('Source data changed during the search')
    report = dict(plan=plan, screen=screened, aggregates=aggregates, selected=selected, tests=tests, baseline_test=baseline_test)
    write(output / 'search.json', report)
    export_summary(output, report)
    print('SELECTED', selected['candidate'], selected['validation_success_rate'], flush=True)


if __name__ == '__main__':
    main()
