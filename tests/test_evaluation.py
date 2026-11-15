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


def test_goal_stall_is_distinct_from_success_and_normal_motion():
    episodes = []
    for name, success, distance, moving in [('stall', False, .032, False), ('moving', False, .032, True),
                                            ('far', False, .3, False), ('success', True, .02, False)]:
        ep = episode(name, success, '' if success else 'timeout', steps=25)
        poses = np.zeros((25, 2))
        if moving: poses[:, 0] = np.linspace(0, 1, 25)
        ep.update(distance=np.full(25, distance), qpos=poses, success_distance=np.array(.03))
        episodes.append(ep)
    result = episode_metrics(episodes)
    assert result['near_goal_stall_count'] == 1 and result['near_goal_stall_rate'] == .25
    assert result['mean_final_distance'] == pytest.approx(np.mean([.032, .032, .3, .02]))
    json.dumps(result)


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

    def load(root, name, config, checkpoint=None):
        loads.append(name)
        return name

    def rollout(env, scene, policy, seed, version):
        success = version.split('.')[0] != 'bc' or scene['scene_id'].endswith('-0')
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


def test_cli_routes_bc_evaluation(evaluation_run, monkeypatch, capsys):
    root, cfg, loads, envs = evaluation_run
    config = root / 'request.json'
    config.write_text(json.dumps(cfg))
    monkeypatch.setattr('sys.argv', ['recap', 'evaluate', '--model', 'bc', '--run', str(root),
                                   '--config', str(config), '--splits', 'id'])
    cli.main()
    assert loads == ['bc']
    assert 'bc/id | success 1/2 (50.0%)' in capsys.readouterr().out
    assert set(json.loads((root / 'evaluation.bc.json').read_text())['bc']) == {'id'}
    assert (root / 'data/id/bc').is_dir()


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


@pytest.mark.parametrize('form', ['bare', 'run_relative', 'absolute'])
def test_specific_checkpoint_only_prints_and_does_not_require_best(evaluation_run, monkeypatch, capsys, form):
    root, cfg, loads, envs = evaluation_run
    (root / 'bc.pt').unlink()
    folder = root / 'checkpoints'
    folder.mkdir()
    checkpoint = folder / 'bc.step00002000.pt'
    checkpoint.touch()
    (root / 'config.json').write_text(json.dumps(cfg))
    (root / 'evaluation.bc.json').write_text('{"previous": true}')
    old = root / 'data/id/bc/previous.npz'
    old.parent.mkdir(parents=True)
    old.write_bytes(b'previous trajectory')
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    argument = checkpoint.name if form == 'bare' else f'checkpoints/{checkpoint.name}' if form == 'run_relative' else str(checkpoint)
    monkeypatch.setattr('sys.argv', ['recap', 'evaluate', '--model', 'bc', '--run', str(root),
                                   '--checkpoint', argument, '--splits', 'id'])
    cli.main()  # Uses the saved config without --config.
    assert loads == ['bc'] and envs[-1].closed
    assert 'bc.step00002000/id | success 1/2 (50.0%)' in capsys.readouterr().out
    after = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert before == after


def test_explicit_best_file_also_does_not_save(evaluation_run):
    root, cfg, loads, envs = evaluation_run
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    report = cli.evaluate_policies(root, cfg, models=['bc'], splits=['id'], checkpoint='bc.pt')
    assert report['bc']['id']['checkpoint'] == str((root / 'bc.pt').resolve())
    assert before == {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_missing_snapshot_fails_without_outputs_or_environment(evaluation_run):
    root, cfg, loads, envs = evaluation_run
    with pytest.raises(FileNotFoundError, match='Checkpoint not found'):
        cli.evaluate_policies(root, cfg, models=['bc'], checkpoint='bc.step00009999.pt')
    assert not loads and not envs and not (root / 'data').exists()


def test_checkpoint_requires_a_single_policy(evaluation_run):
    root, cfg, loads, envs = evaluation_run
    with pytest.raises(ValueError, match='exactly one'):
        cli.evaluate_policies(root, cfg, checkpoint='bc.pt')
    assert not loads and not envs


@pytest.mark.parametrize('arguments', [['evaluate', '--checkpoint', 'bc.pt'], ['train', '--model', 'bc', '--checkpoint', 'bc.pt']])
def test_cli_rejects_checkpoint_without_evaluate_and_model(monkeypatch, arguments):
    monkeypatch.setattr('sys.argv', ['recap', *arguments])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
