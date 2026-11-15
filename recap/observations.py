"""Shared observation layout and conversion of legacy trajectory arrays."""

import numpy as np


OBSERVATION_FIELDS = ('sin_q1', 'sin_q2', 'cos_q1', 'cos_q2',
                      'goal_x', 'goal_y', 'obstacle_x', 'obstacle_y', 'obstacle_radius')
OBSERVATION_DIM = len(OBSERVATION_FIELDS)


def without_velocity(observation):
    observation = np.asarray(observation)
    if observation.ndim and observation.shape[-1] == OBSERVATION_DIM:
        return observation
    if observation.ndim and observation.shape[-1] == 11:
        return np.concatenate((observation[..., :4], observation[..., 6:]), axis=-1)
    raise ValueError('Expected a 9D observation or a legacy 11D observation')


def convert_episode(episode):
    converted = dict(episode)
    for key in ('observation', 'next_observation'):
        if key in converted:
            converted[key] = without_velocity(converted[key])
    return converted
