"""Construct blocked straight-line arm motions for difficulty-guided generation."""

import numpy as np

from .geometry import clearance, points
from .sampling import DEFAULT_SAMPLING


def propose_obstruction(scene, goal_q, difficulty, rng, cfg):
    if difficulty == 'easy':
        return scene
    candidate = dict(scene)
    start = np.asarray(scene['q0'])
    if difficulty == 'hard':
        offset = rng.uniform(.30, .70, size=2) * rng.choice([-1, 1], size=2)
        start = goal_q + offset
        if np.any(np.abs(start) > cfg['env']['joint_limit']):
            return None
        candidate['q0'] = start.tolist()

    # Put the disk inside the swept area of a link along joint-space interpolation.
    middle = start + rng.uniform(.25, .75) * (goal_q - start)
    arm = points(middle)
    link = rng.integers(2)
    direction = arm[link + 1] - arm[link]
    normal = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
    center = arm[link] + rng.uniform(.25, .90) * direction
    center += rng.uniform(-.25, .25) * candidate['radius'] * normal
    bounds = cfg.get('sampling', {}).get('obstacle_center_radius', DEFAULT_SAMPLING['obstacle_center_radius'])
    if not bounds[0] <= np.linalg.norm(center) <= bounds[1]:
        return None
    candidate['obstacle'] = center.tolist()
    for pose in (start, goal_q):
        if clearance(pose, candidate, cfg['env']['link_radius']).min() <= cfg['env']['margin']:
            return None
    return candidate
