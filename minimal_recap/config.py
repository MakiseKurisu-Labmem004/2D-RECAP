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
        if exploration['mode'] not in ['none', 'gaussian'] or len(exploration['std']) != 3 or min(exploration['std']) < 0:
            raise ValueError('Invalid exploration specification')
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
