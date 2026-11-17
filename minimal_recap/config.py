from pathlib import Path
import copy
import hashlib
import json
import os
import random
import subprocess
import numpy as np
import torch
import yaml


def merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load(path):
    path = Path(path).resolve()
    c = yaml.safe_load(path.read_text(encoding='utf-8'))
    parent = c.pop('extends', None)
    c = merge(load(path.parent / parent), c) if parent else c
    if c['environment'].get('task') != 'pedestal_upright' or c['environment'].get('robot') != 'arm3r':
        raise ValueError('Only the arm3r pedestal_upright task is supported')
    if c['reward']['gamma'] != 1 or c['reward']['step'] != -1 or c['reward']['success'] != 0:
        raise ValueError('This experiment contract requires gamma=1, step=-1, success=0')
    if c['expert_rollout']['max_control_steps'] is not None:
        raise ValueError('Expert collection must be unbounded')
    for k in ['policy_rollout', 'evaluation']:
        if c[k]['max_control_steps'] < 1:
            raise ValueError('Policy task budgets must be positive')
    if not 0 <= c['recap']['expert_ratio'] <= 1:
        raise ValueError('expert_ratio must be in [0,1]')
    if c['recap']['n_step'] < 1 or c['evaluation']['diagnostic_prefix_steps'] < 5:
        raise ValueError('Invalid advantage or diagnostic fragment length')
    if c['model']['value_bins'] < 2:
        raise ValueError('At least two value bins required')
    for section in ['bc_training', 'value_training', 'recap']:
        if min(c[section]['epochs'], c[section]['batch_size'], c[section]['updates_per_epoch']) < 1:
            raise ValueError('Training budgets must be positive')
    for section in ['policy_rollout', 'evaluation']:
        exploration = c[section]['exploration']
        if exploration['mode'] not in ['none', 'gaussian'] or len(exploration['std']) != 4 or min(exploration['std']) < 0:
            raise ValueError('Invalid exploration specification')
    if c['environment'].get('task') == 'pedestal_upright':
        e = c['environment']
        if e['lock_rotation']:
            raise ValueError('Cannot lock object rotation in upright task')
        if not 0 < e['overhang'][0] <= e['overhang'][1] < e['block_length']/2:
            raise ValueError('Initial block center of mass must remain on the pedestal')
        if e['max_opening'] <= e['block_thickness'] + 2*e['finger_thickness']:
            raise ValueError('Insufficient gripper opening for side approach')
    if c['environment'].get('gripper_motion') is not None:
        from .gripper import GripperMotion
        e = c['environment']
        if e.get('task') != 'pedestal_upright':
            raise ValueError('Gripper motion governor currently requires pedestal_upright')
        GripperMotion(e['gripper_motion'], e['max_opening'], e['max_opening'], e['timestep'])
        if not np.isfinite(e.get('grip_kv',3)) or e.get('grip_kv',3) <= 0:
            raise ValueError('grip_kv must be positive and finite')
    return c


def seed_all(seed):
    # Configure deterministic CuBLAS before seeding or initializing CUDA.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def fingerprint(c):
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()[:12]


def code_version():
    def git(*args):
        try:
            return subprocess.check_output(['git', *args], stderr=subprocess.DEVNULL).decode().strip()
        except (OSError, subprocess.CalledProcessError):
            return 'unavailable'
    return {'commit': git('rev-parse', 'HEAD'), 'status': git('status', '--porcelain'),
            'source_sha256': hashlib.sha256(b''.join(p.read_bytes() for p in sorted(Path(__file__).parent.glob('*.py')))).hexdigest()}
