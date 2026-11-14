from pathlib import Path

import numpy as np
import pytest
import yaml

from recap.datasets import DATASET_SPLITS, LEGACY_DATASET_SPLITS
from recap.sampling import sample_annulus, sample_scene


@pytest.mark.parametrize('filename', ['default.yaml', 'smoke.yaml'])
def test_dataset_configs(filename):
    cfg = yaml.safe_load((Path('configs') / filename).read_text())
    assert set(DATASET_SPLITS).issubset(cfg['data'])
    assert 'goal_ood' not in cfg['data'] and 'composition_ood' not in cfg['data']
    assert cfg['difficulty_ratio'] == dict(easy=.60, medium=.30, hard=.10)


@pytest.mark.parametrize('split, bounds, large', [
    ('train', [.40, .70], False),
    ('validation', [.40, .70], False),
    ('id', [.40, .70], False),
    ('inner_goal_ood', [.20, .35], False),
    ('outer_goal_ood', [.75, .90], False),
    ('size_ood', [.40, .70], True),
    ('inner_composition_ood', [.20, .35], True),
    ('outer_composition_ood', [.75, .90], True),
])
def test_scene_sampling_covers_four_quadrants(split, bounds, large):
    cfg = yaml.safe_load(Path('configs/default.yaml').read_text())
    rng = np.random.default_rng(42)
    scenes = [sample_scene(rng, split, cfg, 42, i) for i in range(1000)]
    goals = np.array([scene['goal'] for scene in scenes])
    obstacles = np.array([scene['obstacle'] for scene in scenes])
    goal_distance = np.linalg.norm(goals, axis=1)
    obstacle_distance = np.linalg.norm(obstacles, axis=1)
    assert np.all((goal_distance >= bounds[0]) & (goal_distance <= bounds[1]))
    assert np.all((obstacle_distance >= .20) & (obstacle_distance <= .80))
    for positions in (goals, obstacles):
        for x_sign in (-1, 1):
            for y_sign in (-1, 1):
                assert np.sum((positions[:, 0] * x_sign > 0) &
                              (positions[:, 1] * y_sign > 0)) > 150
    radius = np.array([scene['radius'] for scene in scenes])
    low, high = (.10, .13) if large else (.05, .09)
    assert np.all((radius >= low) & (radius <= high))


def test_annulus_is_uniform_in_area():
    rng = np.random.default_rng(42)
    positions = np.array([sample_annulus(rng, [.20, .80]) for _ in range(4000)])
    normalized_area = (np.sum(positions**2, axis=1) - .20**2) / (.80**2 - .20**2)
    assert normalized_area.mean() == pytest.approx(.5, abs=.015)
    assert np.mean(normalized_area < .5) == pytest.approx(.5, abs=.025)


@pytest.mark.parametrize('bounds', [[-.1, .2], [0, 0], [.9, .2],
                                    [np.nan, .2], [0, np.inf], [.2]])
def test_invalid_annulus_bounds(bounds):
    with pytest.raises(ValueError, match='Annulus bounds'):
        sample_annulus(np.random.default_rng(42), bounds)


def test_custom_sampling_and_seed():
    cfg = {'sampling': {'goal_radius': [.45, .50], 'obstacle_center_radius': [.30, .40]}}
    scene = sample_scene(np.random.default_rng(42), 'train', cfg, 42, 0)
    assert scene == sample_scene(np.random.default_rng(42), 'train', cfg, 42, 0)
    assert .45 <= np.linalg.norm(scene['goal']) <= .50
    assert .30 <= np.linalg.norm(scene['obstacle']) <= .40


@pytest.mark.parametrize('split', LEGACY_DATASET_SPLITS)
def test_legacy_dataset_sampling(split):
    cfg = {'data': {name: 1 for name in LEGACY_DATASET_SPLITS}}
    scene = sample_scene(np.random.default_rng(42), split, cfg, 42, 0)
    assert .45 <= np.linalg.norm(scene['goal']) <= .85
    angle = np.arctan2(scene['goal'][1], scene['goal'][0])
    if split in ('goal_ood', 'composition_ood'):
        assert -1.3 <= angle <= -.15
    else:
        assert .15 <= angle <= 1.3
