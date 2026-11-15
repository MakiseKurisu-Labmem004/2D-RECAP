import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from recap.controllers import QuinticExpert, WaypointExpert, make_expert, quintic_reference
from recap.env import ObstacleReach2D
from recap.geometry import points
from recap.planning import rollout
from recap.generation import recover_expert_path, store_expert, write_json
from recap.reexpert import motion_metrics, rebuild_experts
from recap.goal_supervision import goal_anchors


@pytest.fixture
def cfg():
    return yaml.safe_load(Path('configs/default.yaml').read_text())


def test_quintic_endpoints_and_reference_limits(cfg):
    path = np.array([[0., 1.], [.7, -.5], [1., .5]])
    expert = QuinticExpert(path, cfg)
    for index, duration in enumerate(expert.durations):
        start, goal = path[index:index+2]
        for time, expected in ((0., start), (duration, goal)):
            pose, velocity, acceleration = quintic_reference(start, goal, duration, time)
            np.testing.assert_allclose(pose, expected, atol=1e-12)
            np.testing.assert_allclose(velocity, 0., atol=1e-12)
            np.testing.assert_allclose(acceleration, 0., atol=1e-12)
        for time in np.linspace(0, duration, 201):
            pose, velocity, acceleration = quintic_reference(start, goal, duration, time)
            assert abs(velocity).max() <= cfg['planner']['expert_max_velocity'] + 1e-10
            assert abs(acceleration).max() <= cfg['planner']['expert_max_acceleration'] + 1e-10
            assert np.all(pose >= np.minimum(start, goal)-1e-12)
            assert np.all(pose <= np.maximum(start, goal)+1e-12)


def test_legacy_controller_and_invalid_configuration(cfg):
    legacy = copy.deepcopy(cfg)
    legacy['planner'].pop('expert_controller')
    assert isinstance(make_expert(np.array([[0., 1.], [.5, 1.]]), legacy), WaypointExpert)
    cfg['planner']['expert_max_velocity'] = 0
    with pytest.raises(ValueError, match='positive and finite'):
        QuinticExpert(np.array([[0., 1.], [.5, 1.]]), cfg)


def test_real_expert_is_smoother_and_saved_path_is_verifiable(cfg):
    path = np.array([[.2, .8], [.6, .8], [.6, 1.1]])
    scene = dict(scene_id='smooth', split='train', q0=path[0], goal=points(path[-1])[-1], obstacle=[-.5, -.5], radius=.05)
    env = ObstacleReach2D(cfg['env'])
    try:
        old = rollout(env, scene, WaypointExpert(path, cfg), 42, 'expert')
        new = rollout(env, scene, QuinticExpert(path, cfg), 42, 'expert')
    finally:
        env.close()
    assert old['success'][-1] and new['success'][-1]
    before, after = motion_metrics(old, scene, cfg), motion_metrics(new, scene, cfg)
    assert after['peak_joint_jerk'] < before['peak_joint_jerk'] * .5
    assert after['peak_action_jump'] < before['peak_action_jump'] * .5
    assert new['observation'].shape[1] == 9
    assert len(new['distance']) == len(new['action']) == len(new['goal_hold_steps'])
    new.update(expert_path=path, expert_controller=np.array('quintic'))
    np.testing.assert_array_equal(recover_expert_path(new, scene, cfg), path)
    new['action'][0] += .01
    with pytest.raises(ValueError, match='does not match'):
        recover_expert_path(new, scene, cfg)


def test_goal_labels_are_safe_and_point_at_exact_training_goals(cfg):
    scene = dict(scene_id='train', goal=points([.5, 1.])[-1], obstacle=[-.4, -.4], radius=.07)
    cfg['train'].update(goal_anchor_samples=16, goal_anchor_radius=.03)
    observations, actions = goal_anchors([scene], cfg)
    assert observations.shape == (34, 9) and actions.shape == (34, 2)
    poses = np.arctan2(observations[:, :2], observations[:, 2:4])
    for pose, action in zip(poses, actions):
        np.testing.assert_allclose(points(pose+action)[-1], scene['goal'], atol=2e-7)
    np.testing.assert_allclose(observations[:, 4:6], np.tile(scene['goal'], (34, 1)), atol=2e-7)
    again = goal_anchors([scene], cfg)
    np.testing.assert_array_equal(observations, again[0])
    cfg['train']['goal_anchor_samples'] = 0
    assert goal_anchors([scene], cfg)[0].shape == (0, 9)


def test_replay_preserves_scenes_and_stops_at_completed_ik_solution(cfg, tmp_path):
    route = np.array([[-.0055673265, 2.2549972459], [2.2742390626, -1.8631304184]])
    scene = dict(scene_id='other-ik', split='train', q0=route[0].tolist(), goal=points(route[-1])[-1].tolist(),
                 obstacle=[-.6, -.5], radius=.05, metrics=dict(difficulty='easy'))
    legacy = copy.deepcopy(cfg)
    legacy['planner']['expert_controller'] = 'waypoint'
    env = ObstacleReach2D(legacy['env'])
    try:
        previous = rollout(env, scene, WaypointExpert(route, legacy), 42, 'expert')
    finally:
        env.close()
    assert previous['success'][-1]
    previous.update(expert_path=route, expert_controller=np.array('waypoint'))
    source = tmp_path / 'source'
    source.mkdir()
    write_json(source / 'config.json', legacy)
    write_json(source / 'scenes.json', dict(train=[scene]))
    store_expert(source, 'train', scene, previous)
    file = source / 'data/train/expert/expert-42.npz'
    original = file.read_bytes()
    destination = rebuild_experts(source, tmp_path / 'replayed', cfg)
    with np.load(destination / 'data/train/expert/expert-42.npz') as result:
        assert result['success'][-1]
        assert np.linalg.norm(result['expert_path'][-1]-route[0]) < np.linalg.norm(route[-1]-route[0]) * .2
        np.testing.assert_allclose(points(result['expert_path'][-1])[-1], scene['goal'], atol=1e-9)
    saved_scene = json.loads((destination / 'scenes.json').read_text())['train'][0]
    assert saved_scene['metrics'] == scene['metrics']
    assert saved_scene['q0'] == scene['q0'] and saved_scene['goal'] == scene['goal']
    assert file.read_bytes() == original
