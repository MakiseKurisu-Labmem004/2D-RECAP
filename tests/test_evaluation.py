import json

import numpy as np
import pytest

import recap.cli as cli
from recap.evaluation import episode_metrics


def episode(scene_id, success, reason='', steps=2):
    return dict(scene_id=np.array(scene_id), episode_id=np.array(f'bc-{scene_id}'),
                success=np.array([success]), failure_reason=np.array([reason]),
                truncated=np.array([reason == 'timeout']), reward=np.zeros(steps))


def test_success_and_failure_metrics():
    result = episode_metrics([episode('a', True), episode('b', False, 'link1_collision'),
                              episode('c', False, 'timeout')])
    assert result['success_count'] == 1 and result['episodes'] == 3
    assert result['success_rate'] == pytest.approx(1 / 3)
    assert result['collision_rate'] == pytest.approx(1 / 3)
    assert result['timeout_rate'] == pytest.approx(1 / 3)
    assert result['completion_steps'] == 2
    assert 'value_mae' not in result
    assert result['success_ci95'][0] <= result['success_rate'] <= result['success_ci95'][1]


def test_no_success_and_empty_evaluation():
    assert episode_metrics([episode('a', False, 'timeout')])['completion_steps'] is None
    with pytest.raises(ValueError, match='at least one'):
        episode_metrics([])


@pytest.fixture
def evaluation_run(tmp_path, monkeypatch):
    cfg = dict(seed=42, env={})
    scenes = {split: [dict(scene_id=f'{split}-{index}', metrics=dict(difficulty=difficulty))
                       for index, difficulty in enumerate(('easy', 'hard'))]
              for split in ('train', 'validation', 'id', 'inner_goal_ood')}
    (tmp_path / 'scenes.json').write_text(json.dumps(scenes))
    loads = []
    envs = []

    class Env:
        def __init__(self, config):
            self.closed = False
            envs.append(self)

        def close(self):
            self.closed = True

    def load(root, name, config):
        loads.append(name)
        return name

    def rollout(env, scene, policy, seed, version):
        success = version != 'bc' or scene['scene_id'].endswith('-0')
        item = episode(scene['scene_id'], success, '' if success else 'timeout')
        item['episode_id'] = np.array(f'{version}-{seed}')
        return item

    def annotate(episodes, value, config):
        for item in episodes:
            item.update(value_prediction=np.ones(2), mc_return=np.full(2, .75))

    monkeypatch.setattr(cli, 'ObstacleReach2D', Env)
    monkeypatch.setattr(cli, 'load_model', load)
    monkeypatch.setattr(cli, 'policy', lambda model: model)
    monkeypatch.setattr(cli, 'rollout', rollout)
    monkeypatch.setattr(cli, 'annotate', annotate)
    (tmp_path / 'bc.pt').touch()
    return tmp_path, cfg, loads, envs


def test_bc_evaluation_works_with_only_bc_checkpoint(evaluation_run, capsys):
    root, cfg, loads, envs = evaluation_run
    (root / 'evaluation.json').write_text('{"previous": true}')
    report = cli.evaluate_policies(root, cfg, models=['bc'])
    assert loads == ['bc']
    assert set(report['bc']) == {'id', 'inner_goal_ood'}
    assert report['bc']['id']['success_rate'] == .5
    assert report['bc']['id']['by_difficulty']['easy']['success_rate'] == 1
    assert report['bc']['id']['by_difficulty']['hard']['success_rate'] == 0
    assert (root / 'evaluation.json').read_text() == '{"previous": true}'
    assert json.loads((root / 'evaluation.bc.json').read_text()) == report
    assert envs[-1].closed
    assert 'bc overall | success 2/4 (50.0%)' in capsys.readouterr().out


def test_selected_split_does_not_evaluate_other_datasets(evaluation_run):
    root, cfg, loads, envs = evaluation_run
    report = cli.evaluate_policies(root, cfg, models=['bc'], splits=['validation', 'validation'])
    assert set(report['bc']) == {'validation'}
    assert report['bc']['validation']['episodes'] == 2
    assert not (root / 'data/id/bc').exists()


def test_default_full_evaluation_keeps_value_and_gain_metrics(evaluation_run):
    root, cfg, loads, envs = evaluation_run
    for name in ('sft', 'recap', 'value'):
        (root / f'{name}.pt').touch()
    report = cli.evaluate_policies(root, cfg)
    assert loads == ['value', 'bc', 'sft', 'recap']
    assert report['bc']['id']['value_mae'] == .25
    assert report['recap']['id']['gain_over_bc'] == .5
    assert json.loads((root / 'evaluation.json').read_text()) == report
    assert envs[-1].closed


def test_missing_checkpoint_fails_before_creating_environment(evaluation_run):
    root, cfg, loads, envs = evaluation_run
    with pytest.raises(FileNotFoundError):
        cli.evaluate_policies(root, cfg)
    assert not loads and not envs


def test_environment_closes_if_rollout_fails(evaluation_run, monkeypatch):
    root, cfg, loads, envs = evaluation_run
    def fail(*args):
        raise RuntimeError('rollout failed')
    monkeypatch.setattr(cli, 'rollout', fail)
    with pytest.raises(RuntimeError, match='rollout failed'):
        cli.evaluate_policies(root, cfg, models=['bc'])
    assert envs[-1].closed


def test_cli_routes_bc_evaluation(evaluation_run, monkeypatch):
    root, cfg, loads, envs = evaluation_run
    config = root / 'request.json'
    config.write_text(json.dumps(cfg))
    monkeypatch.setattr('sys.argv', ['recap', 'evaluate', '--model', 'bc', '--run', str(root),
                                   '--config', str(config), '--splits', 'id'])
    cli.main()
    assert loads == ['bc']
    assert set(json.loads((root / 'evaluation.bc.json').read_text())['bc']) == {'id'}


def test_cli_train_default_still_selects_bc(evaluation_run, monkeypatch):
    from unittest.mock import Mock
    root, cfg, loads, envs = evaluation_run
    config = root / 'request.json'
    config.write_text(json.dumps(cfg))
    train = Mock()
    monkeypatch.setattr(cli, 'fit', train)
    monkeypatch.setattr('sys.argv', ['recap', 'train', '--run', str(root), '--config', str(config)])
    cli.main()
    train.assert_called_once_with(root, 'bc', cfg, False)
