import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from recap import cli
from recap.configuration import resolve_config, check_data_spec


@pytest.fixture
def cfg():
    result = yaml.safe_load(Path('configs/default.yaml').read_text())
    result['device'] = 'cpu'
    result['train'].update(width=8, depth=1, max_steps=2, validation_steps=1,
                           checkpoint_steps=1, batch_size=2, goal_anchor_samples=0)
    return result


def test_training_uses_requested_hyperparameters_and_frozen_dataset(cfg):
    requested = copy.deepcopy(cfg)
    requested['train'].update(lr=.0005, width=64, max_steps=1000, goal_anchor_samples=64)
    requested['env']['horizon'] += 100
    requested['data']['train'] += 5
    requested['seed'] = 43
    original = copy.deepcopy(cfg)
    effective = resolve_config(cfg, requested, 'train')
    assert effective['env'] == cfg['env'] and effective['data'] == cfg['data']
    assert effective['train'] == requested['train'] and effective['seed'] == 43
    assert cfg == original
    with pytest.raises(ValueError, match='Environment differs'):
        resolve_config(cfg, requested, 'train', explicit=True)


def test_partial_training_config_and_generation_validation(cfg):
    effective = resolve_config(cfg, dict(train=dict(lr=.0005)), 'train', explicit=True)
    assert effective['train']['lr'] == .0005 and effective['env'] == cfg['env']
    assert effective['train']['width'] == cfg['train']['width']
    changed = copy.deepcopy(cfg)
    changed['train']['max_steps'] = 9000
    assert resolve_config(cfg, changed, 'generate', explicit=True) == cfg
    changed['sampling']['goal_radius'] = [.1, .2]
    with pytest.raises(ValueError, match='Generation settings differ'):
        resolve_config(cfg, changed, 'generate', explicit=True)


def test_data_spec_rejects_environment_observation_and_action_changes(tmp_path, cfg):
    check_data_spec(tmp_path, cfg, create=True)
    path = tmp_path / 'data.spec.json'
    original = json.loads(path.read_text())
    for key in ('environment', 'observation_fields', 'action_fields'):
        changed = copy.deepcopy(original)
        changed[key] = {'wrong': True}
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match='definitions differ'):
            check_data_spec(tmp_path, cfg)
    path.write_text(json.dumps(original))
    before = path.read_bytes()
    check_data_spec(tmp_path, cfg)
    assert path.read_bytes() == before


@pytest.fixture
def training_root(tmp_path, cfg, monkeypatch):
    cli.save_json(tmp_path / 'config.json', cfg)
    cli.save_json(tmp_path / 'scenes.json', dict(train=[], validation=[dict(scene_id='v')]))
    ep = dict(observation=np.zeros((4, 9), dtype=np.float32), action=np.zeros((4, 2), dtype=np.float32),
              success=np.array([True]), reward=-np.ones(4), control_dt=np.array(.02))
    for split in ('train', 'validation'):
        directory = tmp_path / 'data' / split / 'expert'
        directory.mkdir(parents=True)
        np.savez_compressed(directory / 'e.npz', **ep)
    class Env:
        def __init__(self, cfg): pass
        def close(self): pass
    monkeypatch.setattr('recap.training.ObstacleReach2D', Env)
    monkeypatch.setattr('recap.training.rollout', lambda *args: dict(success=np.array([True]),
                        truncated=np.array([False]), failure_reason=np.array(['']), reward=np.zeros(1), scene_id=np.array('v')))
    return tmp_path


def test_same_directory_retraining_keeps_dataset_and_independent_configs(training_root, cfg, monkeypatch):
    root = training_root
    frozen = {p: p.read_bytes() for p in [root / 'config.json', root / 'scenes.json', *root.glob('data/*/expert/*.npz')]}
    overrides = root / 'custom.yaml'
    cfg['train']['max_steps'] = 3
    overrides.write_text(yaml.safe_dump(dict(train=cfg['train'], device='cpu')))
    def train(kind, extra=()):
        monkeypatch.setattr('sys.argv', ['recap', 'train', '--run', str(root), '--model', kind,
                                       '--config', str(overrides), *extra])
        cli.main()
    train('bc')
    cfg['train'].update(width=12, lr=.0005, max_steps=2, goal_anchor_samples=0)
    overrides.write_text(yaml.safe_dump(dict(train=cfg['train'], device='cpu')))
    train('bc')
    current = json.loads((root / 'training.bc.json').read_text())
    assert current['train']['width'] == 12 and current['train']['max_steps'] == 2
    assert torch.load(root / 'bc.pt', weights_only=False)['model']['net.0.weight'].shape[0] == 12
    train('value')
    assert (root / 'training.value.json').is_file()
    assert (root / 'training.bc.json').read_text() == json.dumps(current, indent=2)
    for path, content in frozen.items(): assert path.read_bytes() == content
    assert not (root / 'archive').exists()
    # Old snapshots retain their own architecture, independent of current train settings.
    loaded = cli.load_model(root, 'bc', cfg, checkpoint=root / 'checkpoints/bc.step00000003.pt')
    assert loaded.net[0].out_features == 8


def test_resume_uses_checkpoint_settings_and_allows_budget_extension(training_root, cfg, monkeypatch):
    root = training_root
    cli.fit(root, 'bc', cfg)
    old = torch.load(root / 'bc.last.pt', weights_only=False)
    # The dataset config is deliberately different from the trained settings.
    original_config = (root / 'config.json').read_bytes()
    monkeypatch.setattr('sys.argv', ['recap', 'train', '--run', str(root), '--resume'])
    cli.main()
    assert torch.load(root / 'bc.last.pt', weights_only=False)['global_step'] == 2
    path = root / 'extend.yaml'
    path.write_text(yaml.safe_dump(dict(train=dict(max_steps=4))))
    monkeypatch.setattr('sys.argv', ['recap', 'train', '--run', str(root), '--resume', '--train-config', str(path)])
    cli.main()
    result = torch.load(root / 'bc.last.pt', weights_only=False)
    assert result['global_step'] == 4 and result['config']['train']['lr'] == old['config']['train']['lr']
    assert (root / 'config.json').read_bytes() == original_config
    before = (root / 'training.bc.json').read_bytes()
    monkeypatch.setattr('sys.argv', ['recap', 'train', '--run', str(root), '--resume', '--lr', '.0002'])
    with pytest.raises(ValueError, match='checkpoint lr setting'):
        cli.main()
    assert (root / 'training.bc.json').read_bytes() == before


def test_default_checkpoint_environment_is_also_checked(tmp_path, cfg):
    model = cli.Model('bc', cfg['train'], cfg['env']['action_limit'])
    cli.save_checkpoint(tmp_path / 'bc.pt', dict(model=model.state_dict(), config=cfg))
    changed = copy.deepcopy(cfg)
    changed['env']['substeps'] += 1
    with pytest.raises(ValueError, match='environment differs'):
        cli.load_model(tmp_path, 'bc', changed)


def test_lr_override_keeps_saved_environment_even_if_default_changed(tmp_path, cfg, monkeypatch):
    cfg['env']['horizon'] = 123
    cli.save_json(tmp_path / 'config.json', cfg)
    before = (tmp_path / 'config.json').read_bytes()
    captured = []
    monkeypatch.setattr(cli, 'fit', lambda root, kind, config, resume: captured.append(config))
    monkeypatch.setattr('sys.argv', ['recap', 'train', '--run', str(tmp_path), '--lr', '.0005'])
    cli.main()
    assert captured[0]['env']['horizon'] == 123
    assert captured[0]['train']['lr'] == .0005
    assert (tmp_path / 'config.json').read_bytes() == before


def test_resume_rejects_new_supervision_before_rewriting_artifacts(training_root, cfg, monkeypatch):
    root = training_root
    cli.fit(root, 'bc', cfg)
    metadata = root / 'bc.goal_anchors.json'
    metadata.write_bytes(b'original supervision metadata')
    path = root / 'new-supervision.yaml'
    path.write_text(yaml.safe_dump(dict(train=dict(goal_anchor_samples=64))))
    monkeypatch.setattr('sys.argv', ['recap', 'train', '--run', str(root), '--resume', '--train-config', str(path)])
    with pytest.raises(ValueError, match='checkpoint goal_anchor_samples setting'):
        cli.main()
    assert metadata.read_bytes() == b'original supervision metadata'


@pytest.mark.parametrize('invalid', ['action_dimension', 'action_limit', 'control_dt'])
def test_incompatible_trajectories_are_rejected_before_training(training_root, cfg, invalid):
    path = training_root / 'data/train/expert/e.npz'
    with np.load(path) as archive:
        episode = dict(archive)
    if invalid == 'action_dimension': episode['action'] = np.zeros((4, 3))
    if invalid == 'action_limit': episode['action'][:] = 1.
    if invalid == 'control_dt': episode['control_dt'] = np.array(.05)
    np.savez_compressed(path, **episode)
    with pytest.raises(ValueError, match='action definition|control_dt differs'):
        cli.fit(training_root, 'bc', cfg)
    assert not (training_root / 'bc.pt').exists()
    assert not (training_root / 'training.bc.json').exists()
