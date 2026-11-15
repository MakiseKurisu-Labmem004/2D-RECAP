import hashlib
import json

import numpy as np
import pytest

import recap.cli as cli
from recap.generation import store_expert
from recap.migration import migrate_run
from recap.observations import convert_episode, without_velocity


@pytest.mark.parametrize('shape', [(11,), (4, 11), (2, 4, 11)])
def test_legacy_conversion_removes_only_velocity(shape):
    old = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    converted = without_velocity(old)
    np.testing.assert_array_equal(converted, old[..., [0, 1, 2, 3, 6, 7, 8, 9, 10]])
    assert converted.dtype == old.dtype
    np.testing.assert_array_equal(without_velocity(converted), converted)


@pytest.mark.parametrize('observation', [1, [], np.zeros((2, 10)), np.zeros((3, 12))])
def test_unknown_observation_layout_is_rejected(observation):
    with pytest.raises(ValueError, match='9D observation'):
        without_velocity(observation)


@pytest.fixture
def old_run(tmp_path):
    root = tmp_path / 'original'
    folder = root / 'data/train/expert'
    folder.mkdir(parents=True)
    obs = np.arange(33, dtype=np.float32).reshape(3, 11)
    episode = dict(observation=obs, next_observation=obs + 1,
                   action=np.full((3, 2), .05), qpos=np.ones((3, 2)),
                   reward=np.array([-1., -1., 0.]), episode_id=np.array('expert-42'))
    np.savez_compressed(folder / 'expert-42.npz', **episode)
    (root / 'config.json').write_text(json.dumps(dict(seed=42, train=dict(lr=.0003, policy_use_velocity=False))))
    (root / 'scenes.json').write_text(json.dumps(dict(train=[dict(scene_id='train-42')])))
    (root / 'bc.pt').write_bytes(b'old weights')
    return root, episode


def test_episode_loader_and_writers_use_nine_dimensions(old_run, tmp_path):
    source, episode = old_run
    loaded = cli.episodes(source, 'train')[0]
    np.testing.assert_array_equal(loaded['observation'], without_velocity(episode['observation']))
    target = tmp_path / 'new'
    cli.store(target, 'train', 'rollout', episode)
    store_expert(target, 'train', dict(scene_id='train-42'), episode)
    for file in (target / 'data').rglob('*.npz'):
        with np.load(file) as archive:
            assert archive['observation'].shape == (3, 9)
            assert archive['next_observation'].shape == (3, 9)
            np.testing.assert_array_equal(archive['action'], episode['action'])


def test_migration_preserves_source_and_trajectory_alignment(old_run, tmp_path):
    source, episode = old_run
    hashes = {p: hashlib.sha256(p.read_bytes()).digest() for p in source.rglob('*') if p.is_file()}
    target = migrate_run(source, tmp_path / 'converted', lr=.0001)
    with np.load(target / 'data/train/expert/expert-42.npz') as archive:
        for key, value in convert_episode(episode).items():
            np.testing.assert_array_equal(archive[key], value)
    config = json.loads((target / 'config.json').read_text())
    assert config['train'] == dict(lr=.0001)
    assert (source / 'scenes.json').read_bytes() == (target / 'scenes.json').read_bytes()
    assert not (target / 'bc.pt').exists()
    assert json.loads((target / 'migration.json').read_text())['episodes'] == 1
    assert all(hashlib.sha256(p.read_bytes()).digest() == digest for p, digest in hashes.items())


def test_migration_is_idempotent_for_nine_dimensional_data(old_run, tmp_path):
    source, _ = old_run
    first = migrate_run(source, tmp_path / 'first')
    second = migrate_run(first, tmp_path / 'second')
    with np.load(first / 'data/train/expert/expert-42.npz') as a, np.load(second / 'data/train/expert/expert-42.npz') as b:
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])


def test_migration_rejects_overwrite_and_nested_paths(old_run, tmp_path):
    source, _ = old_run
    for path in (source, source / 'nested', tmp_path):
        with pytest.raises(ValueError, match='separate'):
            migrate_run(source, path)
    target = tmp_path / 'existing'
    target.mkdir()
    with pytest.raises(FileExistsError):
        migrate_run(source, target)


@pytest.mark.parametrize('lr', [-1., 0., float('nan'), float('inf')])
def test_invalid_learning_rate_does_not_create_run(old_run, tmp_path, lr):
    source, _ = old_run
    target = tmp_path / 'converted'
    with pytest.raises(ValueError, match='finite and positive'):
        migrate_run(source, target, lr=lr)
    assert not target.exists()


def test_failed_conversion_leaves_no_partial_run(old_run, tmp_path):
    source, episode = old_run
    episode['observation'] = np.zeros((3, 10))
    np.savez_compressed(source / 'data/train/expert/expert-42.npz', **episode)
    with pytest.raises(ValueError, match='9D observation'):
        migrate_run(source, tmp_path / 'converted')
    assert sorted(p.name for p in tmp_path.iterdir()) == ['original']


def test_cli_migrate_uses_saved_configuration(old_run, tmp_path, monkeypatch):
    source, _ = old_run
    target = tmp_path / 'converted'
    monkeypatch.setattr('sys.argv', ['recap', 'migrate', '--run', str(source),
                                   '--output-run', str(target), '--lr', '0.0001'])
    cli.main()
    assert json.loads((target / 'config.json').read_text())['train']['lr'] == .0001


def test_migrate_overrides_only_training_settings(old_run, tmp_path):
    source, _ = old_run
    original = json.loads((source / 'config.json').read_text())
    original['env'] = dict(horizon=123)
    (source / 'config.json').write_text(json.dumps(original))
    result = migrate_run(source, tmp_path / 'adamw', lr=.003,
                         train_overrides=dict(optimizer='adamw', lr=.001, max_steps=10000))
    saved = json.loads((result / 'config.json').read_text())
    assert saved['env'] == dict(horizon=123) and saved['seed'] == 42
    assert saved['train'] == dict(lr=.003, optimizer='adamw', max_steps=10000)
