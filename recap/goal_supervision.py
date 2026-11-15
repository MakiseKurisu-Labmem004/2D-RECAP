"""Safe local expert labels around training goals, used only during BC fitting."""

import numpy as np

from .planning import ik, safe_edge
from .observations import OBSERVATION_DIM


def goal_anchors(scenes, cfg):
    train = cfg['train']
    count = train.get('goal_anchor_samples', 0)
    radius = train.get('goal_anchor_radius', .12)
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError('goal_anchor_samples must be a nonnegative integer')
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('goal_anchor_radius must be positive and finite')
    rng = np.random.default_rng(cfg['seed'] + 40000000)
    observations, actions = [], []
    if count == 0:
        return np.empty((0, OBSERVATION_DIM), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    for scene in scenes:
        for target in ik(np.asarray(scene['goal']), cfg['env']['joint_limit']):
            # Reject obstructed IK branches, including the target pose itself.
            if not safe_edge(target, target, scene, cfg):
                continue
            for offset in np.vstack((np.zeros(2), rng.uniform(-radius, radius, size=(count, 2)))):
                pose = target + offset
                if np.any(abs(pose) > cfg['env']['joint_limit']) or not safe_edge(pose, target, scene, cfg):
                    continue
                observations.append(np.r_[np.sin(pose), np.cos(pose), scene['goal'], scene['obstacle'], scene['radius']])
                actions.append(np.clip(target-pose, -cfg['env']['action_limit'], cfg['env']['action_limit']))
    return (np.asarray(observations, dtype=np.float32).reshape(-1, OBSERVATION_DIM),
            np.asarray(actions, dtype=np.float32).reshape(-1, 2))
