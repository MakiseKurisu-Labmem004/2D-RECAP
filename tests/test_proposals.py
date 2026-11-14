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
        checked += 1
    assert checked >= 10


def test_easy_uses_original_candidate():
    scene = dict(q0=[0., 1.], goal=[.5, .5], obstacle=[-.4, -.3], radius=.07)
    assert propose_obstruction(scene, np.array([.5, 2.]), 'easy',
                               np.random.default_rng(42), {}) is scene
