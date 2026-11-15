"""Compare BC learning rates on validation scenes, then test only the chosen model."""

import argparse
import json
from pathlib import Path
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch

from recap.cli import fit, evaluate_policies
from recap.generation import write_json
from recap.migration import migrate_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='Source run containing expert data and scenes.json')
    parser.add_argument('--output', required=True, help='New directory for all experiments and the comparison report')
    parser.add_argument('--lrs', nargs='+', type=float, default=[3e-4, 1e-3, 3e-3])
    parser.add_argument('--steps', type=int, default=8000)
    parser.add_argument('--interval', type=int, default=500, help='Validation and snapshot interval in optimizer steps')
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if len(set(args.lrs)) != len(args.lrs): parser.error('--lrs must be distinct')
    if args.steps <= 0 or args.interval <= 0 or args.threads <= 0: parser.error('steps, interval, and threads must be positive')
    if not all(np.isfinite(lr) and lr > 0 for lr in args.lrs): parser.error('learning rates must be finite and positive')
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0: parser.error('weight decay must be finite and nonnegative')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    report = dict(source=str(Path(args.run).resolve()), steps=args.steps, interval=args.interval,
                  optimizer='adamw', weight_decay=args.weight_decay,
                  torch_version=torch.__version__, mujoco_version=mujoco.__version__, experiments=[])
    for index, lr in enumerate(args.lrs):
        root = migrate_run(args.run, output / f'{index + 1}-lr{lr:g}', lr=lr,
                           train_overrides=dict(optimizer='adamw', weight_decay=args.weight_decay,
                                                max_steps=args.steps, validation_steps=args.interval, checkpoint_steps=args.interval))
        cfg = json.loads((root / 'config.json').read_text())
        cfg['device'] = args.device
        write_json(root / 'config.json', cfg)
        random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
        fit(root, 'bc', cfg)
        best = torch.load(root / 'bc.pt', map_location='cpu', weights_only=False)
        metrics = best['history'][-1]
        report['experiments'].append(dict(run=str(root), lr=lr, seed=cfg['seed'], best_step=best['global_step'],
                                         validation_success_rate=best['best_success'], validation_mse=best['best'],
                                         validation_by_difficulty=metrics['validation_by_difficulty']))
        write_json(output / 'sweep.json', report)
        print('SWEEP_RESULT', json.dumps(report['experiments'][-1]), flush=True)
    winner = max(report['experiments'], key=lambda row: (row['validation_success_rate'], -row['validation_mse']))
    root = Path(winner['run'])
    cfg = json.loads((root / 'config.json').read_text())
    test = evaluate_policies(root, cfg, models=['bc'], splits=['id'])['bc']['id']
    report.update(selected=winner, id_test=test)
    write_json(output / 'sweep.json', report)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for row in report['experiments']:
        history = json.loads((Path(row['run']) / 'bc.history.json').read_text())
        for axis, key in zip(axes, ('validation', 'validation_success_rate')):
            axis.plot([r['step'] for r in history], [r[key] for r in history], marker='.', label=f'lr={row["lr"]:g}')
    axes[0].set(ylabel='Validation action MSE / action_limit²')
    axes[1].set(ylabel='Validation success rate', ylim=(0, 1))
    for axis in axes:
        axis.set(xlabel='Optimizer updates'); axis.legend(); axis.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output / 'learning_curves.png', dpi=160); plt.close(fig)
    print('SELECTED', json.dumps(dict(lr=winner['lr'], step=winner['best_step'],
          validation=winner['validation_success_rate'], id_success=test['success_rate'], output=str(output))), flush=True)


if __name__ == '__main__':
    main()
