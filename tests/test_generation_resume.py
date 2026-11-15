import hashlib
import json

import numpy as np
import pytest
import yaml

from recap.datasets import DATASET_SPLITS
from recap.geometry import points
import recap.generation as generation


@pytest.fixture
def setup_run(tmp_path, monkeypatch):
    cfg = yaml.safe_load(open('configs/smoke.yaml'))
    cfg['difficulty_ratio'] = dict(easy=1, medium=0, hard=0)
    for split in DATASET_SPLITS:
        cfg['data'][split] = 1
    calls = []
    fail = set()

    class Generator:
        def __init__(self, config):
            self.rejections = {}
            self.difficulty_summary = {}

        def generate(self, split, count, seed, initial_pairs=None):
            calls.append((split, seed))
            if split in fail:
                raise RuntimeError('forced failure')
            pairs = list(initial_pairs or [])
            for index in range(len(pairs), count):
                scene = dict(scene_id=f'{split}-{seed}-{index}', split=split,
                             metrics=dict(difficulty='easy'))
                episode = dict(scene_id=np.array(scene['scene_id']), episode_id=np.array(f'expert-{seed+index}'),
                               seed=np.array(seed+index))
                pairs.append((scene, episode))
                self.rejections['accepted'] = self.rejections.get('accepted', 0) + 1
            self.difficulty_summary[split] = dict(accepted=dict(easy=count, medium=0, hard=0))
            return pairs

    monkeypatch.setattr(generation, 'ScenarioGenerator', Generator)
    return tmp_path, cfg, calls, fail


def test_selected_splits_keep_canonical_seeds(setup_run):
    root, cfg, calls, fail = setup_run
    generation.generate_datasets(root, cfg, selected=['inner_composition_ood', 'outer_composition_ood'])
    assert calls == [('inner_composition_ood', 6000042), ('outer_composition_ood', 7000042)]
    assert set(json.loads((root / 'scenes.json').read_text())) == {'inner_composition_ood', 'outer_composition_ood'}


def test_resume_skips_complete_npz_and_merges_scenes(setup_run):
    root, cfg, calls, fail = setup_run
    generation.generate_datasets(root, cfg, selected=DATASET_SPLITS[:6])
    paths = list((root / 'data').glob('*/*/*.npz'))
    hashes = {path: hashlib.sha256(path.read_bytes()).digest() for path in paths}
    (root / 'scenes.json').unlink()
    calls.clear()
    generation.generate_datasets(root, cfg, resume=True)
    assert calls == [('inner_composition_ood', 6000042), ('outer_composition_ood', 7000042)]
    assert all(hashlib.sha256(path.read_bytes()).digest() == digest for path, digest in hashes.items())
    assert set(json.loads((root / 'scenes.json').read_text())) == set(DATASET_SPLITS)
    assert json.loads((root / 'generation.json').read_text())['accepted'] == 8
    calls.clear()
    generation.generate_datasets(root, cfg, resume=True)
    assert not calls


def test_completed_split_is_checkpointed_before_later_failure(setup_run):
    root, cfg, calls, fail = setup_run
    fail.add('validation')
    with pytest.raises(RuntimeError, match='forced failure'):
        generation.generate_datasets(root, cfg, selected=['train', 'validation'])
    assert set(json.loads((root / 'scenes.json').read_text())) == {'train'}
    assert (root / 'generation.json').exists() and (root / 'difficulty.json').exists()
    fail.clear()
    calls.clear()
    generation.generate_datasets(root, cfg, selected=['train', 'validation'], resume=True)
    assert calls == [('validation', 1000042)]


def test_partial_dataset_is_completed_without_rewriting_existing_episode(setup_run):
    root, cfg, calls, fail = setup_run
    pairs = generation.ScenarioGenerator(cfg).generate('train', 1, 42)
    generation.store_expert(root, 'train', *pairs[0])
    path = root / 'data/train/expert/expert-42.npz'
    before = path.read_bytes()
    cfg['data']['train'] = 2
    calls.clear()
    generation.generate_datasets(root, cfg, selected=['train'], resume=True)
    assert path.read_bytes() == before
    assert len(json.loads((root / 'scenes.json').read_text())['train']) == 2


def test_resume_restores_original_seed_order(setup_run):
    root, cfg, calls, fail = setup_run
    cfg['data']['train'] = 2
    for seed in (90, 100):
        pair = generation.ScenarioGenerator(cfg).generate('train', 1, seed)[0]
        generation.store_expert(root, 'train', *pair)
    generation.generate_datasets(root, cfg, selected=[], resume=True)
    scenes = json.loads((root / 'scenes.json').read_text())['train']
    assert [scene['scene_id'] for scene in scenes] == ['train-90-0', 'train-100-0']


def test_existing_data_requires_explicit_resume(setup_run):
    root, cfg, calls, fail = setup_run
    generation.generate_datasets(root, cfg, selected=['train'])
    with pytest.raises(FileExistsError, match='--resume'):
        generation.generate_datasets(root, cfg)


def test_generation_lock_is_exclusive_and_released(tmp_path):
    with generation.generation_lock(tmp_path):
        with pytest.raises(RuntimeError, match='already running'):
            with generation.generation_lock(tmp_path):
                pass
    with generation.generation_lock(tmp_path):
        pass


def test_failed_json_write_preserves_old_file_and_removes_temporary(tmp_path, monkeypatch):
    path = tmp_path / 'scenes.json'
    path.write_text('{}')
    def fail(*args, **kwargs):
        raise RuntimeError('write failed')
    monkeypatch.setattr(generation.json, 'dump', fail)
    with pytest.raises(RuntimeError, match='write failed'):
        generation.write_json(path, {'train': []})
    assert path.read_text() == '{}'
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('legacy', [False, True])
def test_recover_old_episode_verifies_expert_actions(legacy):
    cfg = yaml.safe_load(open('configs/default.yaml'))
    cfg['env']['action_limit'] = .05  # Configuration saved with the legacy trajectory.
    start = np.array([0., 1.])
    target = np.array([.2, 1.])
    qpos = np.array([[value, 1.] for value in (.04, .08, .12, .16, .2)])
    before = np.vstack([start, qpos[:-1]])
    actions = np.clip(target - before, -.05, .05)
    obs = np.r_[np.sin(start), np.cos(start), [0., 0.], points(target)[-1], [-.5, -.5], .07].astype(np.float32)
    if not legacy:
        obs = np.r_[obs[:4], obs[6:]]
    episode = dict(scene_id=np.array('train-42-0'), episode_id=np.array('expert-42'),
                   observation=np.tile(obs, (5, 1)), applied_command=before + actions,
                   action=actions, qpos=qpos, exploration=np.array(0.),
                   success=np.array([False] * 4 + [True]), reward=np.array([-1.] * 4 + [0.]),
                   clearance=np.full(5, .2))
    scene = generation.recover_scene(episode, 'train', cfg)
    assert scene['metrics']['difficulty'] == 'easy'
    assert scene['recovered_from_expert']
    episode['action'][-1] = [.9, .9]
    with pytest.raises(ValueError, match='Cannot verify'):
        generation.recover_scene(episode, 'train', cfg)
