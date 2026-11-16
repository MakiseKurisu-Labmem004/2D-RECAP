from pathlib import Path
import argparse
import datetime
import importlib.metadata
import hashlib
import json
import shutil
import numpy as np
import torch
import yaml
from .config import load, seed_all, code_version, fingerprint
from .env import PlanarGraspEnv
from .expert import Expert
from .data import atomic_json, collect_episode, episodes, read_episode, freeze_contract, returns
from .learning import load_model, Behavior, train, advantages
from .evaluation import physics_check, evaluate, make_recovery_pool, evaluate_pool, diagnostics


class Runner:
    def __init__(self, c, root):
        self.c, self.root = c, Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.contract = json.loads((self.root/'contract.json').read_text()) if (self.root/'contract.json').exists() else None
        for p in ['expert', 'checkpoints', 'rollouts', 'labels', 'evaluation', 'diagnostics']:
            (self.root/p).mkdir(exist_ok=True)

    def require_contract(self):
        if self.contract is None:
            raise RuntimeError('Run expert and freeze stages first')
        for section, field in [('policy_rollout', 'policy_budget'), ('evaluation', 'evaluation_budget')]:
            if self.c[section]['max_control_steps'] != self.contract[field]:
                raise ValueError('Budget differs from frozen contract')

    def expert(self):
        c, folder = self.c, self.root/'expert'
        existing = episodes(folder)
        successes = sum(e['metadata']['success'] for e in existing)
        env = PlanarGraspEnv(c, None)
        for attempt in range(c['expert_rollout']['max_attempts']):
            if successes >= c['expert_rollout']['successes']:
                break
            path = folder/f'{attempt:06}.npz'
            if path.exists():
                old = read_episode(path)
                if old['metadata']['status'] == 'complete':
                    continue
            else:
                old = None
            expert = Expert(env)
            ep = collect_episode(env, expert, c['seeds']['expert']+attempt,
                                 {'episode_id': f'expert_{attempt}', 'source': 'expert', 'checkpoint': None,
                                  'control_hz': c['environment']['control_hz'], 'data_version': c['dataset']['version']}, path, old)
            successes += ep['metadata']['success']
            print(f'expert {successes}/{c["expert_rollout"]["successes"]}, steps={len(ep["reward"])}', flush=True)
        env.close()
        if successes < c['expert_rollout']['successes']:
            raise RuntimeError('Expert attempt limit reached without requested success count; successes are retained')

    def freeze(self):
        self.contract = freeze_contract(self.c, episodes(self.root/'expert'), self.root/'contract.json')
        lengths = [len(e['reward']) for e in episodes(self.root/'expert') if e['metadata']['success']]
        atomic_json(self.root/'expert_lengths.json', {'min': min(lengths), 'max': max(lengths),
                    'median': float(np.median(lengths)), 'p95': float(np.percentile(lengths, 95)),
                    'budget_note': 'Confirm shared task budgets using these lengths and independent ID validation before formal comparison.'})

    def validation(self, model):
        return evaluate(self.c, model, self.contract, validation=True)

    def bc(self):
        self.require_contract()
        path = self.root/'checkpoints/bc.pt'
        train(self.c, [e for e in episodes(self.root/'expert') if e['metadata']['success']], path, 'bc',
              self.contract, validate=self.validation)

    def rollout(self, round_id=0, checkpoint=None):
        self.require_contract()
        c = self.c
        checkpoint = checkpoint or self.root/'checkpoints/bc.pt'
        model = load_model(checkpoint)
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        folder = self.root/'rollouts'/f'round_{round_id:02}'
        folder.mkdir(exist_ok=True)
        source = 'bc' if round_id == 0 else f'recap_{round_id}'
        atomic_json(folder/'behavior.json', {'checkpoint': str(checkpoint.relative_to(self.root)), 'checkpoint_sha256': checkpoint_hash, 'exploration': c['policy_rollout']['exploration'],
                                          'action_processing': 'network tanh, gaussian optional, normalized clip, actual-position delta servo',
                                          'frozen_during_collection': True, 'round': round_id})
        env = PlanarGraspEnv(c, c['policy_rollout']['max_control_steps'], self.contract['failure_cost'])
        for i in range(c['policy_rollout']['episodes']):
            path = folder/f'{i:06}.npz'
            old = read_episode(path) if path.exists() else None
            if old and old['metadata']['status'] == 'complete':
                continue
            seed = c['seeds']['rollout']+round_id*1000000+i
            behavior = Behavior(model, c['policy_rollout']['exploration'], seed)
            collect_episode(env, behavior, seed, {'episode_id': f'{source}_{i}', 'source': source,
                            'checkpoint': str(checkpoint.relative_to(self.root)), 'checkpoint_sha256': checkpoint_hash, 'exploration': c['policy_rollout']['exploration'],
                            'control_hz': c['environment']['control_hz'], 'data_version': c['dataset']['version'],
                            'reward_version': self.contract['reward_version'], 'contract': self.contract}, path, old)
        env.close()

    def next_rollout(self):
        r = max(0, self.c['recap']['rounds']-1)
        self.rollout(r+1, self.root/'checkpoints'/f'{self.c["recap"]["method"]}_{r:02}.pt')

    def rollout_data(self, round_id=0):
        rounds = range(round_id+1) if self.c['dataset']['mixing'] == 'all_rounds' else [round_id]
        if self.c['dataset']['mixing'] not in ['all_rounds', 'latest_round']:
            raise ValueError('Explicit mixing must be all_rounds or latest_round')
        eps = sum((episodes(self.root/'rollouts'/f'round_{r:02}') for r in rounds), [])
        if not eps:
            raise ValueError('No complete policy rollouts')
        for ep in eps:
            if ep['metadata']['contract'] != self.contract:
                raise ValueError('Mixed reward contracts')
            if len(ep['reward']) > self.contract['policy_budget']:
                raise ValueError('Data exceeds fixed policy horizon')
        return eps

    def value(self, round_id=0):
        path = self.root/'checkpoints'/f'value_{round_id:02}.pt'
        train(self.c, self.rollout_data(round_id), path, 'value', self.contract)

    def label(self, round_id=0):
        model = load_model(self.root/'checkpoints'/f'value_{round_id:02}.pt')
        folder = self.root/'labels'/f'round_{round_id:02}'
        folder.mkdir(exist_ok=True)
        manifest = {}
        for ep in self.rollout_data(round_id):
            with torch.no_grad():
                v = model.value(torch.tensor(ep['observation'], dtype=torch.float32)).numpy()
                nv = model.value(torch.tensor(ep['next_observation'], dtype=torch.float32)).numpy()
            a = advantages(ep, v, nv, self.c['recap']['n_step'])
            key = ep['metadata']['episode_id']
            np.savez_compressed(folder/f'{key}.npz', value=v, returns=returns(ep), advantage=a)
            manifest[key] = {'length': len(a), 'value_checkpoint': str((self.root/'checkpoints'/f'value_{round_id:02}.pt').resolve()),
                             'n_step': self.c['recap']['n_step']}
        atomic_json(folder/'manifest.json', manifest)

    def update(self, round_id=0, method=None):
        method = method or self.c['recap']['method']
        from .config import merge
        c = merge(self.c, {'recap': {'method': method}})
        path = self.root/'checkpoints'/f'{method}_{round_id:02}.pt'
        labels = {}
        for p in (self.root/'labels'/f'round_{round_id:02}').glob('*.npz'):
            with np.load(p) as f:
                labels[p.stem] = f['advantage'].copy()
        # Fixed-dataset comparisons always use identical BC initialization.
        train(c, self.rollout_data(round_id), path, 'update', self.contract,
              initial=load_model(self.root/'checkpoints/bc.pt' if round_id == 0 else self.root/'checkpoints'/f'{method}_{round_id-1:02}.pt'), experts=[e for e in episodes(self.root/'expert') if e['metadata']['success']],
              labels=labels, validate=self.validation)
        return path

    def compare(self):
        self.require_contract()
        paths = {'initial_bc': self.root/'checkpoints/bc.pt'}
        for method in ['aggregate_bc', 'success_bc', 'conditioned', 'unconditioned']:
            paths[method] = self.update(method=method)
        self.evaluate(paths)

    def online(self):
        checkpoint = self.root/'checkpoints/bc.pt'
        for r in range(self.c['recap']['rounds']):
            self.rollout(r, checkpoint)
            self.value(r)
            self.label(r)
            checkpoint = self.update(r)
        # Evaluate the new policy on fresh training scenes too, without updating it.
        self.rollout(self.c['recap']['rounds'], checkpoint)

    def evaluate(self, paths=None):
        self.require_contract()
        if paths is None:
            last = max(0, self.c['recap']['rounds']-1)
            candidate = self.root/'checkpoints'/f'{self.c["recap"]["method"]}_{last:02}.pt'
            paths = {'initial_bc': self.root/'checkpoints/bc.pt'}
            if candidate.exists():
                paths[self.c['recap']['method']] = candidate
        pool_path = self.root/'recovery_pool.json'
        pool = json.loads(pool_path.read_text()) if pool_path.exists() else make_recovery_pool(self.c, self.contract, pool_path)
        result = {}
        for name, path in paths.items():
            model = load_model(path)
            out = self.root/'evaluation'/name
            out.mkdir(exist_ok=True)
            result[name] = {suite: evaluate(self.c, model, self.contract, suite, save_dir=out) for suite in self.c['evaluation']['suites']}
            result[name]['recovery'] = evaluate_pool(self.c, model, self.contract, pool)
            result[name]['checkpoint'] = str(path.resolve())
        atomic_json(self.root/'metrics.json', result)

    def diagnose(self):
        r = max(0, self.c['recap']['rounds']-1)
        behavior = json.loads((self.root/'rollouts'/f'round_{r:02}'/'behavior.json').read_text())
        reference = Path(behavior['checkpoint'])
        reference = reference if reference.is_absolute() else self.root/reference
        diagnostics(self.c, load_model(reference), load_model(self.root/'checkpoints'/f'value_{r:02}.pt'),
                    self.contract, self.root/'diagnostics')

    def execute(self, stage):
        if stage == 'physics':
            physics_check(self.c, self.root/'physics.json')
        elif stage in ['expert', 'freeze', 'bc', 'rollout', 'value', 'label', 'update', 'evaluate', 'diagnose', 'online', 'compare', 'next_rollout']:
            getattr(self, stage)()
        else:
            raise ValueError(f'Unknown stage: {stage}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--reuse', type=Path, help='Reuse immutable expert, contract, BC and round-0 rollout artifacts for matched comparisons')
    parser.add_argument('--stages', nargs='+')
    args = parser.parse_args()
    c = load(args.config)
    seed_all(c['seeds']['training'])
    root = args.resume or Path(c['run']['output'])/(datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+c['run']['name']+'_'+fingerprint(c))
    root.mkdir(parents=True, exist_ok=True)
    resolved = root/'resolved.yaml'
    if resolved.exists() and yaml.safe_load(resolved.read_text(encoding='utf-8')) != c:
        raise ValueError('Resume requires identical resolved configuration; use --stages to choose work')
    resolved.write_text(yaml.safe_dump(c, sort_keys=False), encoding='utf-8')
    if args.reuse and not (root/'reuse.json').exists():
        old_c = yaml.safe_load((args.reuse/'resolved.yaml').read_text(encoding='utf-8'))
        for key in ['environment', 'observation', 'reward', 'model', 'seeds', 'policy_rollout', 'evaluation', 'dataset', 'bc_training']:
            if old_c[key] != c[key]:
                raise ValueError(f'Reuse mismatch: {key}')
        shutil.copytree(args.reuse/'expert', root/'expert', dirs_exist_ok=True)
        (root/'checkpoints').mkdir(exist_ok=True)
        for source in (args.reuse/'checkpoints').glob('bc.pt*'):
            shutil.copy2(source, root/'checkpoints'/source.name)
        shutil.copytree(args.reuse/'rollouts/round_00', root/'rollouts/round_00', dirs_exist_ok=True)
        shutil.copy2(args.reuse/'contract.json', root/'contract.json')
        atomic_json(root/'reuse.json', {'source': str(args.reuse.resolve())})
    manifest = {'code': code_version(), 'config_hash': fingerprint(c), 'seed': c['seeds'],
                'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'mujoco', 'torch', 'gymnasium', 'PyYAML']}}
    if not (root/'manifest.json').exists():
        atomic_json(root/'manifest.json', manifest)
    print(f'run directory: {root.resolve()}', flush=True)
    atomic_json(root/('execution_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json'), manifest)
    runner = Runner(c, root)
    for stage in args.stages or c['run']['stages']:
        marker = root/f'{stage}.done.json'
        if marker.exists():
            continue
        print(f'stage: {stage}', flush=True)
        runner.execute(stage)
        atomic_json(marker, {'completed': True, 'code': code_version()})
    print('completed', flush=True)


if __name__ == '__main__':
    main()
