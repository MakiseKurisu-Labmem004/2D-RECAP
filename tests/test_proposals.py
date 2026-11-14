import numpy as np
import pytest

from recap.geometry import clearance, points
from recap.proposals import propose_obstruction


@pytest.mark.parametrize('difficulty', ['medium', 'hard'])
def test_constructed_obstacle_blocks_direct_motion(difficulty):
    rng = np.random.default_rng(42)
    cfg = dict(env=dict(joint_limit=3., link_radius=.015, margin=.008))
    goal_q = np.array([.5, 2.])
    checked = 0
    for _ in range(300):
        original = dict(q0=rng.uniform([-2., -.8], [2., 2.4]).tolist(),
                        goal=points(goal_q)[-1].tolist(), obstacle=[-.4, -.3], radius=.07)
        scene = propose_obstruction(original, goal_q, difficulty, rng, cfg)
        if scene is None:
            continue
        assert original['obstacle'] == [-.4, -.3]
        assert scene['goal'] == original['goal']
        start = np.array(scene['q0'])
        assert np.all(np.abs(start) <= 3.)
        assert .20 <= np.linalg.norm(scene['obstacle']) <= .80
        assert clearance(start, scene, .015).min() > .008
        assert clearance(goal_q, scene, .015).min() > .008
        gaps = [clearance(q, scene, .015).min() for q in np.linspace(start, goal_q, 151)]
        assert min(gaps) < 0
        if difficulty == 'hard':
            assert np.all(np.abs(start - goal_q) >= .30)
            assert np.all(np.abs(start - goal_q) <= .70)
        else:
            assert np.all(np.abs(start - goal_q) >= .60)
            assert np.all(np.abs(start - goal_q) <= 1.10)
        checked += 1
    assert checked >= 10


def test_easy_uses_original_candidate():
    scene = dict(q0=[0., 1.], goal=[.5, .5], obstacle=[-.4, -.3], radius=.07)
    assert propose_obstruction(scene, np.array([.5, 2.]), 'easy',
                               np.random.default_rng(42), {}) is scene


def test_inner_large_obstacle_has_feasible_hard_constructions():
    from recap.sampling import sample_scene
    from pathlib import Path
    import yaml
    import recap.planning as planning

    cfg = yaml.safe_load(Path('configs/default.yaml').read_text())
    rng = np.random.default_rng(6000042)
    valid = 0
    for attempt in range(300):
        scene = sample_scene(rng, 'inner_composition_ood', cfg, 6000042, attempt)
        goals = planning.ik(np.array(scene['goal']), cfg['env']['joint_limit'])
        if not goals:
            continue
        goal = goals[rng.integers(len(goals))]
        candidate = propose_obstruction(scene, goal, 'hard', rng, cfg)
        if candidate is None:
            continue
        assert .20 <= np.linalg.norm(candidate['goal']) <= .35
        assert .10 <= candidate['radius'] <= .13
        assert clearance(candidate['q0'], candidate, .015).min() > .008
        assert clearance(goal, candidate, .015).min() > .008
        assert not planning.safe_edge(np.array(candidate['q0']), goal, candidate, cfg)
        valid += 1
    assert valid >= 15
