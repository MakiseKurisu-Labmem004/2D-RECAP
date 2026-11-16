import numpy as np
import pytest

from recap.models import returns
from recap.rewards import REWARD_VERSION, check_value_rewards, relabel_rewards, return_scale


@pytest.mark.parametrize('horizon', [1, 60, 500])
def test_success_outranks_earliest_failure_and_timeout_normalizes(horizon):
    latest_success = -(horizon - 1)
    earliest_failure = -horizon
    assert latest_success > earliest_failure
    rewards = np.full(horizon, -1.)
    rewards[-1] = -horizon
    assert returns(rewards, return_scale(horizon))[0] == -1


def test_legacy_failure_relabel_preserves_source_arrays():
    rewards = np.array([-1., -150.])
    ep = dict(reward=rewards, terminated=np.array([False, True]),
              truncated=np.array([False, False]), success=np.array([False, False]))
    relabel_rewards(ep, 500)
    np.testing.assert_array_equal(ep['reward'], [-1., -500.])
    np.testing.assert_array_equal(rewards, [-1., -150.])
    relabel_rewards(ep, 500)
    np.testing.assert_array_equal(ep['reward'], [-1., -500.])


def test_reward_checkpoint_version_is_required():
    with pytest.raises(ValueError, match='Retrain'):
        check_value_rewards({'config': {'train': {}}})
    check_value_rewards({'config': {'train': {'reward_version': REWARD_VERSION}}})


def test_value_fit_uses_new_targets_without_rewriting_dataset(tmp_path, monkeypatch):
    from pathlib import Path
    import yaml
    from recap import cli
    cfg = yaml.safe_load(Path('configs/default.yaml').read_text())
    cfg['device'] = 'cpu'
    ep = dict(observation=np.zeros((2, 9), dtype=np.float32), action=np.zeros((2, 2)),
              reward=np.array([-1., -150.]), terminated=np.array([False, True]),
              truncated=np.zeros(2, dtype=bool), success=np.zeros(2, dtype=bool), episode_id=np.array('legacy'))
    for split in ('train', 'validation'):
        cli.store(tmp_path, split, 'expert', ep)
    before = {p: p.read_bytes() for p in tmp_path.rglob('*.npz')}
    captured = []
    monkeypatch.setattr(cli, 'train_steps', lambda *args: captured.append(args))
    cli.fit(tmp_path, 'value', cfg)
    args = captured[0]
    assert args[2]['train']['reward_version'] == REWARD_VERSION
    np.testing.assert_allclose(args[5][1].numpy(), [-501/999, -500/999])
    assert all(p.read_bytes() == content for p, content in before.items())
