import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

import recap.cli as cli
import recap.training as training
from recap.models import Model


@pytest.fixture
def config():
    cfg = yaml.safe_load(Path('configs/default.yaml').read_text())
    cfg['device'] = 'cpu'
    cfg['train'].update(width=8, depth=1, epochs=2, max_steps=None, batch_size=4, lr=0., goal_anchor_samples=0)
    return cfg


@pytest.mark.parametrize('kind', ['bc', 'sft', 'recap', 'value'])
def test_all_models_use_nine_observations(config, kind):
    model = Model(kind, config['train'], .05)
    assert model.net[0].in_features == (12 if kind == 'recap' else 9)
    assert model(torch.randn(4, 9)).shape[0] == 4
    with pytest.raises(ValueError, match='9D observations'):
        model(torch.randn(4, 11))


def test_loading_uses_checkpoint_architecture(tmp_path, config):
    saved = copy.deepcopy(config)
    saved['env']['action_limit'] = .05
    model = Model('bc', saved['train'], .05)
    torch.save(dict(model=model.state_dict(), config=saved), tmp_path / 'bc.pt')
    runtime = copy.deepcopy(config)
    runtime['train'].update(width=32)
    loaded = cli.load_model(tmp_path, 'bc', runtime)
    x = torch.randn(3, 9)
    torch.testing.assert_close(loaded(x), model(x))


def test_specific_checkpoint_loads_selected_weights_and_step(tmp_path, config):
    folder = tmp_path / 'checkpoints'
    folder.mkdir()
    model = Model('bc', config['train'], config['env']['action_limit'])
    checkpoint = folder / 'bc.step00002000.pt'
    torch.save(dict(model=model.state_dict(), config=config, global_step=2000, model_kind='bc'), checkpoint)
    loaded = cli.load_model(tmp_path, 'bc', config, checkpoint=checkpoint)
    assert loaded.checkpoint_info['step'] == 2000
    x = torch.randn(3, 9)
    torch.testing.assert_close(loaded(x), model(x))
    changed = copy.deepcopy(config)
    changed['env']['success_distance'] = .05
    with pytest.raises(ValueError, match='environment differs'):
        cli.load_model(tmp_path, 'bc', changed, checkpoint=checkpoint)
    torch.save(dict(model=model.state_dict(), config=config, model_kind='sft'), checkpoint)
    with pytest.raises(ValueError, match='contains sft'):
        cli.load_model(tmp_path, 'bc', config, checkpoint=checkpoint)


@pytest.mark.parametrize('kind', ['bc', 'sft', 'recap', 'value'])
def test_old_checkpoint_requires_retraining(tmp_path, config, kind):
    weights = Model(kind, config['train'], .05).state_dict()
    weights['net.0.weight'] = torch.zeros(config['train']['width'], 14 if kind == 'recap' else 11)
    torch.save(dict(model=weights, config=config), tmp_path / f'{kind}.pt')
    with pytest.raises(ValueError, match='Retrain with 9D'):
        cli.load_model(tmp_path, kind, config)


@pytest.fixture
def training_run(tmp_path, monkeypatch):
    ep = dict(observation=np.zeros((4, 9), dtype=np.float32),
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
    monkeypatch.setattr(training, 'ObstacleReach2D', Env)
    outcomes = iter([True, False, True, False])
    monkeypatch.setattr(cli, 'rollout', lambda *args: dict(success=np.array([next(outcomes)]),
                       truncated=np.array([False]), reward=np.zeros(1), failure_reason=np.array(['']), scene_id=np.array('v0')))
    monkeypatch.setattr(training, 'rollout', lambda *args: cli.rollout(*args))
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
    assert best['model']['net.0.weight'].shape[1] == 9
    assert best['config']['train']['validation_rollout_interval'] == 10
    assert config == original
    assert all(env.closed for env in envs)

    # Resume restores effective settings even when the run config omits them.
    config['train']['epochs'] = 3
    cli.fit(root, 'bc', config, resume=True)
    resumed = torch.load(root / 'bc.last.pt', weights_only=False)
    assert resumed['epoch'] == 2
    assert resumed['config']['train']['validation_rollout_interval'] == 10
    assert resumed['model']['net.0.weight'].shape[1] == 9


def test_old_resume_requires_retraining(training_run, config):
    root, envs = training_run
    cli.fit(root, 'bc', config)
    ck = torch.load(root / 'bc.last.pt', weights_only=False)
    ck['model']['net.0.weight'] = torch.zeros(config['train']['width'], 11)
    torch.save(ck, root / 'bc.last.pt')
    config['train']['epochs'] = 3
    with pytest.raises(ValueError, match='Retrain with 9D'):
        cli.fit(root, 'bc', config, resume=True)


def test_resume_rejects_changed_selection_mode(training_run, config):
    root, _ = training_run
    cli.fit(root, 'bc', config)
    config['train']['validation_rollout_interval'] = 0
    with pytest.raises(ValueError, match='validation_rollout_interval'):
        cli.fit(root, 'bc', config, resume=True)


def test_explicit_mse_selection(training_run, config):
    root, envs = training_run
    config['train'].update(validation_rollout_interval=0)
    (root / 'scenes.json').unlink()
    cli.fit(root, 'bc', config)
    ck = torch.load(root / 'bc.pt', weights_only=False)
    assert ck['model']['net.0.weight'].shape[1] == 9
    assert not envs


def test_sft_initializes_from_nine_dimensional_bc(training_run, config):
    root, envs = training_run
    saved = copy.deepcopy(config)
    bc = Model('bc', saved['train'], .05)
    torch.save(dict(model=bc.state_dict(), config=saved), root / 'bc.pt')
    cli.fit(root, 'sft', config)
    assert cli.load_model(root, 'sft', config).net[0].in_features == 9
    assert not envs


def test_validation_environment_closes_on_failure(training_run, config, monkeypatch):
    root, envs = training_run
    def fail(*args):
        raise RuntimeError('validation failed')
    monkeypatch.setattr(cli, 'rollout', fail)
    with pytest.raises(RuntimeError, match='validation failed'):
        cli.fit(root, 'bc', config)
    assert envs[-1].closed


def test_step_budget_and_snapshot_selection(training_run, config):
    root, envs = training_run
    config['train'].update(max_steps=5, validation_steps=3, checkpoint_steps=2)
    cli.fit(root, 'bc', config)
    best = torch.load(root / 'bc.pt', weights_only=False)
    last = torch.load(root / 'bc.last.pt', weights_only=False)
    assert last['global_step'] == 5
    assert [row['step'] for row in last['history']] == [1, 3, 5]
    assert best['global_step'] in (1, 5)
    assert last['config']['train']['optimizer'] == 'adamw'
    assert last['optimizer']['param_groups'][0]['weight_decay'] == 1e-4
    assert sorted(p.name for p in (root / 'checkpoints').iterdir()) == [
        'bc.step00000002.pt', 'bc.step00000004.pt', 'bc.step00000005.pt']
    assert all(env.closed for env in envs)


def test_exact_step_resume_matches_uninterrupted_training(training_run, config, monkeypatch):
    root, _ = training_run
    # Five samples give a short last batch. Step 2 is inside a shuffled epoch.
    x = np.arange(45, dtype=np.float32).reshape(5, 9) / 45
    ep = dict(observation=x, action=np.full((5, 2), .02, dtype=np.float32), success=np.array([True]))
    monkeypatch.setattr(cli, 'episodes', lambda *args: [ep.copy()])
    monkeypatch.setattr(cli, 'rollout', lambda *args: dict(success=np.array([True]),
                       truncated=np.array([False]), reward=np.zeros(1), failure_reason=np.array(['']), scene_id=np.array('v0')))
    cfg = copy.deepcopy(config)
    cfg['train'].update(max_steps=5, validation_steps=2, checkpoint_steps=2, lr=.001, batch_size=2)
    torch.manual_seed(42)
    cli.fit(root, 'bc', cfg)
    expected = torch.load(root / 'bc.last.pt', weights_only=False)
    partial = torch.load(root / 'checkpoints/bc.step00000002.pt', weights_only=False)
    torch.save(partial, root / 'bc.last.pt')
    torch.manual_seed(999)
    cli.fit(root, 'bc', cfg, resume=True)
    actual = torch.load(root / 'bc.last.pt', weights_only=False)
    assert actual['global_step'] == 5
    assert actual['history'] == expected['history']
    for key in expected['model']:
        torch.testing.assert_close(actual['model'][key], expected['model'][key], rtol=0, atol=0)


def test_optimizer_resume_rejects_changing_algorithm(training_run, config):
    root, _ = training_run
    cli.fit(root, 'bc', config)
    config['train']['optimizer'] = 'adam'
    with pytest.raises(ValueError, match='checkpoint optimizer'):
        cli.fit(root, 'bc', config, resume=True)


def test_failed_checkpoint_write_preserves_previous_state(tmp_path, monkeypatch):
    path = tmp_path / 'bc.last.pt'
    training.save_checkpoint(path, dict(step=123))
    before = path.read_bytes()
    def fail(*args):
        raise RuntimeError('disk error')
    monkeypatch.setattr(training.torch, 'save', fail)
    with pytest.raises(RuntimeError, match='disk error'):
        training.save_checkpoint(path, dict(step=124))
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_legacy_optimizer_resume_keeps_adam_without_decay(config):
    model = Model('bc', config['train'], .05)
    tc = dict(lr=.0003)
    optimizer = training.make_optimizer(model, tc, saved=dict(lr=.0003))
    assert type(optimizer) is torch.optim.Adam
    assert optimizer.param_groups[0]['weight_decay'] == 0.
