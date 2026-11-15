from pathlib import Path

import numpy as np
import pytest
import yaml

from recap.geometry import points,clearance
from recap.splines import JointSpline, UnsafeSplineError, derivative, evaluate, curve_clear
from recap.control_filter import C3CommandFilter
from recap.controllers import SplineExpert
from recap.env import ObstacleReach2D
from recap.planning import rollout
from recap.evaluation import episode_metrics
from recap.generation import recover_expert_path,recover_scene


@pytest.fixture
def cfg():
    return yaml.safe_load(Path('configs/default.yaml').read_text())


def scene_for(path):
    return dict(scene_id='spline', split='train', q0=path[0].tolist(), goal=points(path[-1])[-1].tolist(),
                obstacle=[-.5, -.5], radius=.05)


def test_global_spline_is_c3_and_moves_through_internal_waypoint(cfg):
    path = np.array([[.2, .6], [.6, .6], [.6, 1.1]])
    spline = JointSpline(path, cfg, scene_for(path))
    assert spline.blend == 1.
    for index in range(len(spline.spans)-1):
        for order in range(4):
            left = evaluate(derivative(spline.coefficients[index], order), 1.) / (spline.spans[index]*spline.duration)**order
            right = evaluate(derivative(spline.coefficients[index+1], order), 0.) / (spline.spans[index+1]*spline.duration)**order
            np.testing.assert_allclose(left, right, atol=2e-7)
    knot = spline.knots[1]*spline.duration
    assert np.linalg.norm(spline.reference(knot, 1)) > .01
    for time, pose in ((0., path[0]), (spline.duration, path[-1])):
        np.testing.assert_allclose(spline.reference(time), pose, atol=1e-9)
        for order in (1, 2, 3): np.testing.assert_allclose(spline.reference(time, order), 0., atol=2e-7)
    for time in np.linspace(0., spline.duration, 501):
        for order, name in ((1, 'velocity'), (2, 'acceleration'), (3, 'jerk')):
            assert abs(spline.reference(time, order)).max() <= cfg['planner'][f'expert_max_{name}'] + 1e-7


def test_curve_check_detects_collision_between_safe_endpoints(cfg):
    path = np.array([[0., 0.], [.4, 0.]])
    scene = dict(obstacle=[.45, .09], radius=.04)
    # Both endpoint configurations are clear; the arm crosses the disk between them.
    control = np.linspace(path[0], path[-1], 8)
    assert not curve_clear(control, scene, cfg)
    with pytest.raises(UnsafeSplineError):
        JointSpline(path, cfg, scene)


def test_narrow_scene_reduces_spline_derivatives_until_entire_curve_is_clear(cfg):
    path = np.array([[-1.0229723006910585,-2.471756292450392],[-.8882681564245809,-2.4301675977653634],
                     [-.8547486033519553,-2.363128491620112],[-.8882681564245809,-2.229050279329609],
                     [-1.3910614525139664,-1.893854748603352],[-1.6776503222675978,-1.7736461922096352]])
    scene = dict(obstacle=[-.25449058413505554,-.33642110228538513],radius=.05753593519330025)
    spline = JointSpline(path,cfg,scene)
    assert 0 <= spline.blend < 1
    for time in np.linspace(0,spline.duration,301):
        pose=spline.reference(time)
        assert abs(pose).max() <= cfg['env']['joint_limit']+1e-9
        assert clearance(pose,scene,cfg['env']['link_radius']).min() > cfg['env']['margin']


@pytest.mark.parametrize('key,value', [('substeps',0),('horizon',0),('hold',0),('timestep',0.),('action_limit',0.)])
def test_invalid_control_settings_fail_before_simulation(cfg,key,value):
    cfg['env'][key]=value
    with pytest.raises(ValueError,match=key):
        ObstacleReach2D(cfg['env'])


def test_filter_exact_response_monotone_bounds_and_c3_at_target_switch():
    tau, duration = .003, .005
    filtered = C3CommandFilter([0., 0.], tau)
    result = filtered.advance(np.ones(2), duration)
    x = duration/tau
    expected = 1-np.exp(-x)*(1+x+x*x/2+x*x*x/6)
    np.testing.assert_allclose(result, expected, atol=1e-14)
    before = filtered.derivatives()
    filtered.advance(np.array([-1., 2.]), 0.)
    for old, new in zip(before, filtered.derivatives()): np.testing.assert_allclose(old, new, atol=1e-7)
    for target in (np.array([3., -3.]), np.array([-3., 3.])):
        for _ in range(30):
            assert abs(filtered.advance(target, duration)).max() <= 3.+1e-10
    once, split = C3CommandFilter([.2, -.1], tau), C3CommandFilter([.2, -.1], tau)
    expected = once.advance(np.ones(2), .01)
    split.advance(np.ones(2), .004)
    np.testing.assert_allclose(split.advance(np.ones(2), .006), expected, atol=1e-14)
    tiny = C3CommandFilter([0., 0.], 1e-8)
    np.testing.assert_array_equal(tiny.advance(np.ones(2), .005), np.ones(2))
    with pytest.raises(ValueError, match='nonnegative'):
        tiny.advance(np.ones(2), -.01)


def test_substep_commands_are_saved_and_reset_for_all_policies(cfg):
    path = np.array([[.2, .6], [.6, .6], [.6, 1.1]])
    env = ObstacleReach2D(cfg['env'])
    scene = scene_for(path)
    try:
        env.reset(options=dict(scene=scene))
        _, _, _, _, info = env.step(np.array([.03, -.03]))
        assert info['control_commands'].shape == (cfg['env']['substeps'], 2)
        assert np.linalg.norm(info['control_commands'][0]-info['applied_command']) > 1e-4
        first = info['control_commands'].copy()
        env.reset(options=dict(scene=scene))
        np.testing.assert_allclose(env.step(np.array([.03, -.03]))[-1]['control_commands'], first)
        expert = SplineExpert(path, cfg, scene)
        episode = rollout(env, scene, expert, 42, 'expert')
        assert episode['success'][-1]
        assert episode['control_commands'].shape[1:] == (cfg['env']['substeps'], 2)
        np.testing.assert_allclose(episode['substep_qpos'][-1, int(episode['executed_substeps'][-1])-1], episode['qpos'][-1])
        assert env.metadata['render_fps'] == 50
        assert episode['observation'].shape[1] == 9
        episode.update(expert_path=path, expert_controller=np.array('spline'))
        np.testing.assert_array_equal(recover_expert_path(episode, scene, cfg), path)
        recovered = recover_scene(episode, 'train', cfg)
        assert recovered['metrics']['joint_path_length'] == pytest.approx(expert.trajectory.length, abs=1e-8)
    finally:
        env.close()


def test_expanded_goal_radius_keeps_hold_rule_and_strict_score(cfg):
    path = np.array([[.4, 1.], [.4, 1.]])
    scene = scene_for(path)
    scene['goal'] = (points(path[0])[-1] + [.032, 0.]).tolist()
    env = ObstacleReach2D(cfg['env'])
    try:
        episode = rollout(env, scene, lambda obs, env: np.zeros(2), 42, 'bc')
        assert episode['success'][-1] and len(episode['action']) == cfg['env']['hold']
        report = episode_metrics([episode])
        assert report['success_rate'] == 1.
        assert report['strict_success_rate'] == 0.
        assert report['tolerance_only_success_count'] == 1
        assert report['completion_seconds'] == pytest.approx(cfg['env']['hold']*.02)
        scene['goal'] = (points(path[0])[-1] + [.029, 0.]).tolist()
        strict = rollout(env, scene, lambda obs, env: np.zeros(2), 42, 'bc')
        assert episode_metrics([strict])['strict_success_rate'] == 1.
    finally:
        env.close()


def test_recover_spline_start_when_first_command_hits_joint_limit(cfg):
    path = np.array([[2.99999, 1.], [2.999995, 1.]])
    scene = scene_for(path)
    env = ObstacleReach2D(cfg['env'])
    try:
        episode = rollout(env, scene, SplineExpert(path, cfg, scene), 42, 'expert')
    finally:
        env.close()
    episode.update(expert_path=path, expert_controller=np.array('spline'))
    recovered = recover_scene(episode, 'train', cfg)
    np.testing.assert_allclose(recovered['q0'], path[0], atol=2e-6)


def test_goal_hold_resets_outside_radius_and_old_control_mode_is_preserved(cfg):
    path = np.array([[.4, 1.], [.4, 1.]])
    scene = scene_for(path)
    center = points(path[0])[-1]
    scene['goal'] = (center+[.032, 0.]).tolist()
    env = ObstacleReach2D(cfg['env'])
    try:
        env.reset(options=dict(scene=scene))
        for _ in range(cfg['env']['hold']-1): assert not env.step(np.zeros(2))[2]
        env.scene['goal'] = (center+[.04, 0.]).tolist()
        assert not env.step(np.zeros(2))[2] and env.held == 0
        env.scene['goal'] = (center+[.032, 0.]).tolist()
        for _ in range(cfg['env']['hold']-1): assert not env.step(np.zeros(2))[2]
        assert env.step(np.zeros(2))[2]
    finally:
        env.close()
    legacy = dict(cfg['env'])
    legacy.pop('control_interpolation')
    env = ObstacleReach2D(legacy)
    try:
        env.reset(options=dict(scene=scene))
        info = env.step(np.array([.02, -.01]))[-1]
        np.testing.assert_allclose(info['control_commands'], np.tile(info['applied_command'], (legacy['substeps'], 1)))
    finally:
        env.close()
