"""Success-first finite-horizon rewards and compatibility for stored episodes."""

import numpy as np


REWARD_VERSION = 'failure_horizon_v1'


def return_scale(horizon):
    # H-1 ordinary steps followed by the failure penalty -H.
    return 2 * horizon - 1


def relabel_rewards(episode, horizon):
    """Update rewards in memory; preserve recorded observations/actions on disk."""
    rewards = np.asarray(episode['reward'], dtype=np.float64).copy()
    # Both legacy and current trajectories use -1/0 except terminal failures.
    failure = rewards < -1
    if all(key in episode for key in ('terminated', 'truncated', 'success')):
        terminal = np.asarray(episode['terminated'], dtype=bool) | np.asarray(episode['truncated'], dtype=bool)
        failure = failure | (terminal & ~np.asarray(episode['success'], dtype=bool))
    rewards[failure] = -float(horizon)
    episode['reward'] = rewards
    episode['reward_version'] = np.array(REWARD_VERSION)


def check_value_rewards(checkpoint):
    if checkpoint.get('config', {}).get('train', {}).get('reward_version') != REWARD_VERSION:
        raise ValueError('Value/RECAP reward definition changed to failure=-horizon. '
                         'Retrain value and recap without --resume; existing trajectories and BC can be reused.')
