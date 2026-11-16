from pathlib import Path
import json
import os
import numpy as np


def json_numpy(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False, default=json_numpy), encoding='utf-8')
    os.replace(temp, path)


def write_episode(path, ep):
    arrays = {k: np.asarray(ep[k]) for k in ['observation', 'action', 'reward', 'next_observation', 'terminated', 'truncated']}
    arrays['metadata'] = np.array(json.dumps(ep['metadata'], allow_nan=False, default=json_numpy))
    arrays['states'] = np.array(json.dumps(ep['states'], allow_nan=False, default=json_numpy))
    arrays['controller_states'] = np.array(json.dumps(ep.get('controller_states', []), allow_nan=False, default=json_numpy))
    temp = Path(str(path)+'.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(temp, path)


def read_episode(path):
    with np.load(path, allow_pickle=False) as f:
        ep = {k: f[k].copy() for k in f.files if k not in ['metadata', 'states', 'controller_states']}
        for k in ['metadata', 'states', 'controller_states']:
            ep[k] = json.loads(str(f[k]))
    return ep


def episodes(folder, complete=True):
    result = [read_episode(p) for p in sorted(Path(folder).glob('*.npz'))]
    return [e for e in result if e['metadata']['status'] == 'complete'] if complete else result


def returns(ep):
    if ep['metadata']['status'] != 'complete' or not len(ep['reward']) or not ep['terminated'][-1] or np.asarray(ep['truncated']).any():
        raise ValueError('Monte Carlo requires a complete task trajectory, never an external truncation')
    r = np.cumsum(ep['reward'][::-1], dtype=np.float64)[::-1].copy()
    if not np.isfinite(r).all():
        raise ValueError('Nonfinite returns')
    return r


def split(eps, fraction, seed):
    if len(eps) < 2:
        raise ValueError('Need at least 2 complete episodes for trajectory-level split')
    order = np.random.default_rng(seed).permutation(len(eps))
    n = min(len(eps)-1, max(1, round(len(eps)*fraction)))
    return [eps[i] for i in order[n:]], [eps[i] for i in order[:n]]


def freeze_contract(c, expert, path):
    lengths = [len(e['reward']) for e in expert if e['metadata']['success']]
    if not lengths:
        raise ValueError('Collect successful experts before freezing reward')
    M = max(max(lengths), c['policy_rollout']['max_control_steps'], c['evaluation']['max_control_steps'])
    cost = M if c['reward']['failure_cost'] == 'auto' else c['reward']['failure_cost']
    if cost <= M-1:
        raise ValueError('Fastest failure must be strictly worse than slowest guaranteed success')
    low = -(max(c['policy_rollout']['max_control_steps'], c['evaluation']['max_control_steps'])-1+cost)
    support = [low, 0] if c['reward']['support'] == 'auto' else c['reward']['support']
    if support[0] > low or support[1] < 0:
        raise ValueError('Value support does not cover guaranteed returns')
    contract = {'M': M, 'failure_cost': cost, 'support': support, 'reward_version': c['reward']['version'],
                'dataset_version': c['dataset']['version'], 'expert_max_length': max(lengths),
                'policy_budget': c['policy_rollout']['max_control_steps'], 'evaluation_budget': c['evaluation']['max_control_steps']}
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError('Frozen contract changed: create a new experiment version and recompute all returns')
    atomic_json(path, contract)
    return contract


def collect_episode(env, policy, seed, metadata, path=None, resume=None):
    if resume:
        ep = resume
        obs = env.restore(ep['states'][-1])
        if hasattr(policy, 'load_state_dict'):
            policy.load_state_dict(ep['controller_states'][-1])
        if hasattr(policy, 'restore_rng') and 'behavior_rng' in ep['metadata']:
            policy.restore_rng(ep['metadata']['behavior_rng'])
    else:
        obs, _ = env.reset(seed=seed)
        ep = {k: [] for k in ['observation', 'action', 'reward', 'next_observation', 'terminated', 'truncated']}
        ep.update(metadata=dict(metadata, seed=seed, scene=env.scene, status='active', success=False),
                  states=[env.snapshot()], controller_states=[])
    try:
        while not env.done:
            if hasattr(policy, 'state_dict') and not ep['controller_states']:
                ep['controller_states'].append(policy.state_dict())
            action = policy(obs)
            nxt, r, term, trunc, info = env.step(action)
            for k, v in zip(['observation', 'action', 'reward', 'next_observation', 'terminated', 'truncated'],
                            [obs, env.executed_action.copy(), r if np.isfinite(r) else 0., nxt, term, trunc]):
                ep[k] = list(ep[k])
                ep[k].append(v)
            ep['states'].append(env.snapshot())
            if hasattr(policy, 'state_dict'):
                ep['controller_states'].append(policy.state_dict())
            obs = nxt
        ep['metadata'].update(info, status='complete', length=env.steps)
    except (Exception, KeyboardInterrupt) as ex:
        ep['metadata'].update(status='aborted', reason=type(ex).__name__+': '+str(ex))
        if hasattr(policy, 'rng'):
            ep['metadata']['behavior_rng'] = policy.rng.bit_generator.state
        if path:
            write_episode(path, ep)
        raise
    if path:
        write_episode(path, ep)
    return ep
