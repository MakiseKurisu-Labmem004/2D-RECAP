"""Optimizer configuration and exact update-budget training with resumable snapshots."""

import torch
from pathlib import Path
from tempfile import NamedTemporaryFile
from tqdm import tqdm

from .generation import write_json
from .env import ObstacleReach2D
from .planning import rollout
from .evaluation import episode_metrics, difficulty_metrics


def save_checkpoint(path, checkpoint):
    path = Path(path)
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(checkpoint, stream)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def make_optimizer(model, config, saved=None):
    legacy = saved is not None
    defaults = dict(optimizer='adam' if legacy else 'adamw', weight_decay=0. if legacy else 1e-4)
    for key, default in defaults.items():
        previous = saved.get(key, default) if legacy else default
        if legacy and key in config and config[key] != previous:
            raise ValueError(f'Resume with the checkpoint {key} setting')
        config.setdefault(key, previous)
    optimizers = dict(adam=torch.optim.Adam, adamw=torch.optim.AdamW)
    if config['optimizer'] not in optimizers:
        raise ValueError('optimizer must be adam or adamw')
    return optimizers[config['optimizer']](model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])


def positive_integer(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def train_steps(root, kind, cfg, model, optimizer, train, validation, loss, scenes, policy, checkpoint=None):
    tc = cfg['train']
    budget = positive_integer(tc['max_steps'], 'max_steps')
    saved_train = checkpoint.get('config', {}).get('train', {}) if checkpoint else {}
    validation_steps = positive_integer(tc.get('validation_steps', saved_train.get('validation_steps', 500)), 'validation_steps')
    checkpoint_steps = positive_integer(tc.get('checkpoint_steps', saved_train.get('checkpoint_steps', validation_steps)), 'checkpoint_steps')
    if checkpoint and validation_steps != saved_train.get('validation_steps', validation_steps):
        raise ValueError('Resume with the checkpoint validation_steps setting')
    tc.update(validation_steps=validation_steps, checkpoint_steps=checkpoint_steps)
    x, y, labels = train
    vx, vy, vl = validation
    state = checkpoint.get('train_state') if checkpoint else None
    if checkpoint and state is None:
        raise ValueError('Step training requires a step-training checkpoint; start a new run')
    step = state['step'] if state else 0
    epoch = state['epoch'] if state else 0
    permutation = state['permutation'].cpu() if state else torch.empty(0, dtype=torch.long)
    offset = state['offset'] if state else 0
    total = state['loss_sum'] if state else 0.
    samples = state['samples'] if state else 0
    history = list(checkpoint['history']) if checkpoint else []
    best = checkpoint['best'] if checkpoint else float('inf')
    best_success = checkpoint.get('best_success', -1.) if checkpoint else -1.
    if state and state['transitions'] != len(x):
        raise ValueError('Resume requires unchanged training data')
    snapshots = root / 'checkpoints'
    snapshots.mkdir(exist_ok=True)
    tqdm.write(f'{kind}: {budget} optimizer steps; {tc["optimizer"]}, lr={tc["lr"]}, '
               f'weight_decay={tc["weight_decay"]}; validate every {validation_steps} steps')
    with tqdm(total=budget, initial=step, desc=f'Train {kind}', unit='step', dynamic_ncols=True) as progress:
        while step < budget:
            if offset == len(permutation):
                if len(permutation): epoch += 1
                permutation = torch.randperm(len(x))
                offset = 0
            ids = permutation[offset:offset + tc['batch_size']].to(x.device)
            model.train()
            optimizer.zero_grad()
            err = loss(x[ids], y[ids], labels[ids], True)
            err.backward()
            optimizer.step()
            total += err.item() * len(ids)
            samples += len(ids)
            offset += len(ids)
            step += 1
            progress.update(1)
            evaluate = step == 1 or step % validation_steps == 0 or step == budget
            snapshot = step % checkpoint_steps == 0 or step == budget
            improved = False
            if evaluate:
                model.eval()
                with torch.no_grad(): error = loss(vx, vy, vl, False).item()
                record = dict(step=step, epoch=epoch, train=total / samples, validation=error)
                total = 0.
                samples = 0
                improved = error < best
                if kind == 'bc':
                    env = ObstacleReach2D(cfg['env'])
                    try:
                        with tqdm(scenes, desc='BC validation rollout', unit='episode', leave=False, dynamic_ncols=True) as episodes:
                            results = [rollout(env, scene, policy(model), cfg['seed'] + 30000000 + i, 'bc-validation')
                                       for i, scene in enumerate(episodes)]
                    finally:
                        env.close()
                    metrics = episode_metrics(results)
                    success = metrics['success_rate']
                    record.update(validation_success_rate=success,
                                  validation_collision_rate=metrics['collision_rate'],
                                  validation_timeout_rate=metrics['timeout_rate'],
                                  validation_by_difficulty=difficulty_metrics(results, scenes))
                    if 'strict_success_rate' in metrics:
                        record.update(validation_strict_success_rate=metrics['strict_success_rate'],
                                      validation_tolerance_only_success_count=metrics['tolerance_only_success_count'])
                    improved = success > best_success or (success == best_success and error < best)
                    if improved: best_success = success
                if improved: best = error
                history.append(record)
                write_json(root / f'{kind}.history.json', history)
                postfix = dict(train=f'{record["train"]:.5f}', validation=f'{error:.5f}')
                if kind == 'bc': postfix['best_success'] = f'{best_success:.1%}'
                progress.set_postfix(postfix)
                tqdm.write(f'{kind} step={step}, train={record["train"]:.6f}, validation={error:.6f}'
                           + (f', success={record["validation_success_rate"]:.1%}, best={best_success:.1%}' if kind == 'bc' else ''))
            if evaluate or snapshot:
                train_state = dict(step=step, epoch=epoch, permutation=permutation, offset=offset,
                                   loss_sum=total, samples=samples, transitions=len(x))
                ck = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch, global_step=step,
                          train_state=train_state, best=best, best_success=best_success, history=history,
                          rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                          config=cfg)
                save_checkpoint(root / f'{kind}.last.pt', ck)
                if improved: save_checkpoint(root / f'{kind}.pt', ck)
                if snapshot: save_checkpoint(snapshots / f'{kind}.step{step:08d}.pt', ck)
    return history
