from pathlib import Path
from unittest.mock import Mock

import numpy as np
import yaml

from recap.geometry import points
import recap.planning as planning
from recap.proposals import propose_obstruction


def test_local_detours_are_safe_and_below_hard_threshold():
    cfg = yaml.safe_load(Path('configs/default.yaml').read_text())
    rng = np.random.default_rng(42)
    goal = np.array([.5, 2.])
    found = 0
    for _ in range(100):
        original = dict(q0=[0., 0.], goal=points(goal)[-1].tolist(), obstacle=[0., 0.], radius=.07)
        scene = propose_obstruction(original, goal, 'medium', rng, cfg)
        if scene is None:
            continue
        start = np.array(scene['q0'])
        path = planning.local_medium_detour(start, goal, scene, cfg)
        if path is None:
            continue
        assert not planning.safe_edge(start, goal, scene, cfg)
        assert all(planning.safe_edge(a, b, scene, cfg) for a, b in zip(path[:-1], path[1:]))
        ratio = np.linalg.norm(np.diff(path, axis=0), axis=1).sum() / np.linalg.norm(goal - start)
        assert ratio < cfg['planner']['hard_ratio']
        found += 1
    assert found >= 5


def test_local_search_has_bounded_work(monkeypatch):
    cfg = yaml.safe_load(Path('configs/default.yaml').read_text())
    check = Mock(return_value=False)
    monkeypatch.setattr(planning, 'safe_edge', check)
    assert planning.local_medium_detour(np.zeros(2), np.ones(2), {}, cfg) is None
    assert check.call_count == 6


def test_local_search_respects_custom_hard_threshold(monkeypatch):
    cfg = yaml.safe_load(Path('configs/default.yaml').read_text())
    cfg['planner']['hard_ratio'] = 1.01
    check = Mock(return_value=True)
    monkeypatch.setattr(planning, 'safe_edge', check)
    assert planning.local_medium_detour(np.zeros(2), np.ones(2), {}, cfg) is None
    check.assert_not_called()


def test_zero_length_motion_has_no_local_detour():
    assert planning.local_medium_detour(np.zeros(2), np.zeros(2), {}, {}) is None
