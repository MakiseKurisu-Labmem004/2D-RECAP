"""Frozen dataset settings and independent per-model training settings."""

import copy
import json
from pathlib import Path

from .generation import write_json
from .observations import OBSERVATION_FIELDS


def dataset_settings(cfg):
    frozen = copy.deepcopy({key: value for key, value in cfg.items() if key not in ('train', 'device')})
    for key in ('rollouts', 'exploration'):
        frozen.get('data', {}).pop(key, None)
    return frozen


def check_environment(saved, requested):
    if 'env' in requested and saved['env'] != requested['env']:
        raise ValueError('Environment differs from the frozen dataset: keep its environment, observation and action definitions')


def data_spec(cfg):
    return dict(environment=cfg['env'], observation_fields=list(OBSERVATION_FIELDS),
                action_fields=['joint1_target_offset', 'joint2_target_offset'])


def check_data_spec(root, cfg, *, create=False):
    path = Path(root) / 'data.spec.json'
    expected = data_spec(cfg)
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != expected:
            raise ValueError('Dataset environment/observation/action definitions differ from data.spec.json')
    elif create:
        write_json(path, expected)


def resolve_config(saved, requested, command, *, explicit=False, checkpoint=None, environment_explicit=None):
    """Use saved geometry/physics; train requests replace only train/seed/device."""
    if saved is None:
        if 'env' not in requested:
            raise ValueError('A new run requires a full configuration containing env')
        return copy.deepcopy(requested)
    if environment_explicit is None:
        environment_explicit = explicit
    if environment_explicit:
        check_environment(saved, requested)
    if command in ('generate', 'all') and explicit:
        candidate = copy.deepcopy(saved)
        candidate.update(requested)
        if dataset_settings(saved) != dataset_settings(candidate):
            raise ValueError('Generation settings differ from the frozen dataset; use its saved settings or a new dataset')
    result = copy.deepcopy(saved)
    if command in ('train', 'all'):
        if checkpoint is not None:
            check_environment(saved, checkpoint)
            result['train'] = copy.deepcopy(checkpoint['train'])
            result['seed'] = checkpoint.get('seed', saved['seed'])
            result['device'] = checkpoint.get('device', saved['device'])
        # An implicit resume uses its checkpoint, even if defaults changed.
        if checkpoint is None or explicit:
            result['train'].update(copy.deepcopy(requested.get('train', {})))
            result['seed'] = requested.get('seed', result['seed'])
            result['device'] = requested.get('device', result['device'])
    elif explicit:
        result['device'] = requested.get('device', result['device'])
        if command == 'rollout':
            for key in ('rollouts', 'exploration'):
                if key in requested.get('data', {}):
                    result['data'][key] = requested['data'][key]
    return result


def check_resume(cfg, previous, progress):
    check_environment(previous, cfg)
    old = previous.get('train', {})
    # Budgets can grow and snapshot frequency can change. Changing optimizer,
    # data supervision, batching or model settings requires fresh training.
    allowed = {'max_steps', 'epochs', 'checkpoint_steps'}
    for key, value in cfg['train'].items():
        if key not in allowed and key in old and value != old[key]:
            raise ValueError(f'Resume with the checkpoint {key} setting; omit --resume to retrain on the same data')
    if cfg.get('seed') != previous.get('seed', cfg.get('seed')):
        raise ValueError('Resume with the checkpoint seed')
    budget = cfg['train'].get('max_steps')
    if budget is not None and budget < progress:
        raise ValueError('max_steps cannot be smaller than the checkpoint progress')


def save_training_config(root, kind, cfg):
    write_json(Path(root) / f'training.{kind}.json',
               dict(seed=cfg['seed'], device=cfg['device'], train=cfg['train']))
