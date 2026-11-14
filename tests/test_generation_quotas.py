from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from tqdm import tqdm
import yaml

import recap.planning as planning


@pytest.fixture
def controlled_generator(monkeypatch):
    cfg = yaml.safe_load(Path('configs/default.yaml').read_text())
    sequence = ['easy'] * 8 + ['medium'] * 4 + ['hard']
    envs = []

    class FakeEnv:
        def __init__(self, config):
            self.closed = False
            envs.append(self)

        def close(self):
            self.closed = True

    def sample_scene(rng, split, config, seed, attempt):
        return dict(q0=[0., 0.], goal=[.5, .5], obstacle=[.5, .2], radius=.07,
                    scene_id=f'{split}-{attempt}', candidate_difficulty=sequence[attempt % len(sequence)])

    def plan(start, goal, scene, config):
        if scene['candidate_difficulty'] == 'hard':
            return np.array([start, [-1., 0.], goal])
        return np.array([start, goal])

    def rollout(env, scene, policy, seed, version, **kwargs):
        gap = .01 if scene['candidate_difficulty'] == 'hard' else .05
        return dict(success=np.array([True]), reward=np.array([0.]),
                    clearance=np.array([gap]), qpos=np.array([[0., 0.]]))

    monkeypatch.setattr(planning, 'ObstacleReach2D', FakeEnv)
    monkeypatch.setattr(planning, 'sample_scene', sample_scene)
    monkeypatch.setattr(planning, 'clearance', lambda *args: np.array([1., 1.]))
    monkeypatch.setattr(planning, 'ik', lambda *args: [np.array([1., 0.])])
    monkeypatch.setattr(planning, 'plan', plan)
    monkeypatch.setattr(planning, 'rollout', rollout)
    monkeypatch.setattr(planning, 'safe_edge',
                        lambda start, goal, scene, config: scene['candidate_difficulty'] == 'easy')
    monkeypatch.setattr(planning, 'tqdm', lambda *args, **kwargs: tqdm(*args, **kwargs, disable=True))
    cfg['planner']['max_attempts'] = len(sequence)
    return cfg, sequence, envs


def test_generation_fills_exact_quotas(controlled_generator):
    cfg, sequence, envs = controlled_generator
    generator = planning.ScenarioGenerator(cfg)
    pairs = generator.generate('train', 10, 42)
    counts = Counter(scene['metrics']['difficulty'] for scene, episode in pairs)
    assert counts == dict(easy=6, medium=3, hard=1)
    assert generator.rejections['difficulty_quota_full'] == 3
    assert generator.difficulty_summary['train'] == dict(
        target=dict(easy=6, medium=3, hard=1), accepted=dict(easy=6, medium=3, hard=1), attempts=13)
    assert envs[-1].closed


def test_missing_difficulty_fails_without_filling_with_easy(controlled_generator):
    cfg, sequence, envs = controlled_generator
    sequence[:] = ['easy']
    generator = planning.ScenarioGenerator(cfg)
    with pytest.raises(RuntimeError, match="missing difficulty quotas:.*'medium': 3, 'hard': 1"):
        generator.generate('train', 10, 42)
    assert generator.difficulty_summary['train']['accepted'] == dict(easy=6, medium=0, hard=0)
    assert envs[-1].closed


def test_quotas_are_independent_for_each_split(controlled_generator):
    cfg, sequence, envs = controlled_generator
    generator = planning.ScenarioGenerator(cfg)
    for split in ('train', 'id'):
        assert len(generator.generate(split, 10, 42)) == 10
        assert generator.difficulty_summary[split]['accepted'] == dict(easy=6, medium=3, hard=1)
    assert all(env.closed for env in envs)


def test_second_ik_branch_can_fill_a_missing_quota(controlled_generator, monkeypatch):
    cfg, sequence, envs = controlled_generator
    sequence[:] = ['easy']
    cfg['difficulty_ratio'] = dict(easy=.5, medium=.5, hard=0)
    cfg['planner']['max_attempts'] = 2
    monkeypatch.setattr(planning, 'ik', lambda *args: [np.array([1., 0.]), np.array([2., 0.])])
    monkeypatch.setattr(planning, 'safe_edge', lambda start, goal, scene, config: goal[0] == 1)
    pairs = planning.ScenarioGenerator(cfg).generate('train', 2, 42)
    assert Counter(scene['metrics']['difficulty'] for scene, episode in pairs) == dict(easy=1, medium=1)


def test_config_without_ratios_preserves_unrestricted_sampling(controlled_generator):
    cfg, sequence, envs = controlled_generator
    cfg.pop('difficulty_ratio')
    sequence[:] = ['easy']
    generator = planning.ScenarioGenerator(cfg)
    assert len(generator.generate('train', 5, 42)) == 5
    assert generator.difficulty_summary['train']['target'] is None


def test_invalid_ratio_fails_before_creating_environment(controlled_generator):
    cfg, sequence, envs = controlled_generator
    cfg['difficulty_ratio']['easy'] = .9
    with pytest.raises(ValueError, match='sum to 1'):
        planning.ScenarioGenerator(cfg).generate('train', 10, 42)
    assert not envs


def test_zero_scene_count(controlled_generator):
    cfg, sequence, envs = controlled_generator
    cfg['planner']['max_attempts'] = 0
    generator = planning.ScenarioGenerator(cfg)
    assert generator.generate('train', 0, 42) == []
    assert generator.difficulty_summary['train']['target'] == dict(easy=0, medium=0, hard=0)
    assert envs[-1].closed
