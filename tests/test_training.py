import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

import recap.cli as cli
from recap.models import Model


@pytest.fixture
def config():
    cfg = yaml.safe_load(Path('configs/default.yaml').read_text())
    cfg['device'] = 'cpu'
    cfg['train'].update(width=8, depth=1, epochs=2, batch_size=4, lr=0.)
    return cfg


def test_policy_velocity_mask_and_value_state(config):
    cfg = dict(config['train'], policy_use_velocity=False)
    x = torch.randn(4, 11)
    changed = x.clone()
    changed[:, 4:6] += 10
    for kind in ('bc', 'sft', 'recap'):
        model = Model(kind, cfg, .05)
        torch.testing.assert_close(model.action(x), model.action(changed), rtol=0, atol=0)
    value = Model('value', cfg, .05)
    assert value.use_velocity
    assert not torch.equal(value(x), value(changed))
    assert Model('bc', config['train'], .05).use_velocity  # Legacy default.


@pytest.mark.parametrize('legacy', [False, True])
def test_loading_uses_checkpoint_features_and_architecture(tmp_path, config, legacy):
    saved = copy.deepcopy(config)
    if not legacy:
        saved['train']['policy_use_velocity'] = False
    model = Model('bc', saved['train'], .05)
    torch.save(dict(model=model.state_dict(), config=saved), tmp_path / 'bc.pt')
    runtime = copy.deepcopy(config)
    runtime['train'].update(width=32, policy_use_velocity=legacy)
    loaded = cli.load_model(tmp_path, 'bc', runtime)
    assert loaded.use_velocity is legacy
    x = torch.randn(3, 11)
    torch.testing.assert_close(loaded(x), model(x))


@pytest.fixture
def training_run(tmp_path, monkeypatch):
    ep = dict(observation=np.zeros((4, 11), dtype=np.float32),
              action=np.full((4, 2), .025, dtype=np.float32), success=np.array([True]))
    monkeypatch.setattr(cli, 'episodes', lambda *args: [ep.copy()])
    (tmp_path / 'scenes.json').write_text(json.dumps(dict(validation=[dict(scene_id='v0')])))
    environments = []

    class Env:
        def __init__(self, cfg):
            self.closed = False
            environments.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(cli, 'ObstacleReach2D', Env)
    outcomes = iter([True, False, True, False])
    monkeypatch.setattr(cli, 'rollout', lambda *args: dict(success=np.array([next(outcomes)])))
    return tmp_path, environments


def test_best_checkpoint_uses_closed_loop_success(training_run, config):
    root, envs = training_run
    config['train']['lr'] = .001
    original = copy.deepcopy(config)
    cli.fit(root, 'bc', config)
    best = torch.load(root / 'bc.pt', weights_only=False)
    last = torch.load(root / 'bc.last.pt', weights_only=False)
    assert best['epoch'] == 0 and last['epoch'] == 1
    assert last['history'][-1]['validation'] < best['history'][-1]['validation']
    assert [row['validation_success_rate'] for row in last['history']] == [1., 0.]
    assert not best['config']['train']['policy_use_velocity']
    assert best['config']['train']['validation_rollout_interval'] == 10
    assert config == original
    assert all(env.closed for env in envs)

    # Resume restores effective settings even when the run config omits them.
    config['train']['epochs'] = 3
    cli.fit(root, 'bc', config, resume=True)
    resumed = torch.load(root / 'bc.last.pt', weights_only=False)
    assert resumed['epoch'] == 2
    assert resumed['config']['train']['validation_rollout_interval'] == 10
    assert not resumed['config']['train']['policy_use_velocity']


def test_legacy_resume_preserves_velocity_and_loss_selection(training_run, config):
    root, envs = training_run
    cli.fit(root, 'bc', config)
    ck = torch.load(root / 'bc.last.pt', weights_only=False)
    ck['config']['train'].pop('policy_use_velocity')
    ck['config']['train'].pop('validation_rollout_interval')
    ck.pop('best_success')
    torch.save(ck, root / 'bc.last.pt')
    config['train']['epochs'] = 3
    count = len(envs)
    cli.fit(root, 'bc', config, resume=True)
    resumed = torch.load(root / 'bc.last.pt', weights_only=False)
    assert resumed['config']['train']['policy_use_velocity']
    assert resumed['config']['train']['validation_rollout_interval'] == 0
    assert 'validation_success_rate' not in resumed['history'][-1]
    assert len(envs) == count


def test_resume_rejects_changed_feature_mode(training_run, config):
    root, _ = training_run
    cli.fit(root, 'bc', config)
    config['train']['policy_use_velocity'] = True
    with pytest.raises(ValueError, match='policy_use_velocity'):
        cli.fit(root, 'bc', config, resume=True)


def test_resume_rejects_changed_selection_mode(training_run, config):
    root, _ = training_run
    cli.fit(root, 'bc', config)
    config['train']['validation_rollout_interval'] = 0
    with pytest.raises(ValueError, match='validation_rollout_interval'):
        cli.fit(root, 'bc', config, resume=True)


def test_explicit_velocity_and_mse_selection(training_run, config):
    root, envs = training_run
    config['train'].update(policy_use_velocity=True, validation_rollout_interval=0)
    (root / 'scenes.json').unlink()
    cli.fit(root, 'bc', config)
    ck = torch.load(root / 'bc.pt', weights_only=False)
    assert ck['config']['train']['policy_use_velocity']
    assert not envs


@pytest.mark.parametrize('use_velocity', [False, True])
def test_sft_inherits_bc_features(training_run, config, use_velocity):
    root, envs = training_run
    saved = copy.deepcopy(config)
    saved['train']['policy_use_velocity'] = use_velocity
    bc = Model('bc', saved['train'], .05)
    torch.save(dict(model=bc.state_dict(), config=saved), root / 'bc.pt')
    cli.fit(root, 'sft', config)
    assert cli.load_model(root, 'sft', config).use_velocity is use_velocity
    assert not envs


def test_validation_environment_closes_on_failure(training_run, config, monkeypatch):
    root, envs = training_run
    def fail(*args):
        raise RuntimeError('validation failed')
    monkeypatch.setattr(cli, 'rollout', fail)
    with pytest.raises(RuntimeError, match='validation failed'):
        cli.fit(root, 'bc', config)
    assert envs[-1].closed
