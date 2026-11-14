"""Construct blocked straight-line arm motions for difficulty-guided generation."""

import numpy as np

from .geometry import clearance, points
from .sampling import DEFAULT_SAMPLING


def propose_obstruction(scene, goal_q, difficulty, rng, cfg):
    if difficulty == 'easy':
        return scene
    candidate = dict(scene)
    # Bound the required motion so a local detour can fit the expert step budget.
    span = (.60, 1.10) if difficulty == 'medium' else (.30, .70)
    offset = rng.uniform(*span, size=2) * rng.choice([-1, 1], size=2)
    start = goal_q + offset
    if difficulty == 'medium':
        offset = np.where(np.abs(start) > cfg['env']['joint_limit'], -offset, offset)
        start = goal_q + offset
    if np.any(np.abs(start) > cfg['env']['joint_limit']):
        return None
    candidate['q0'] = start.tolist()

    centers = []
    if difficulty == 'medium':
        # Check a bounded set of shallow placements before starting a planner.
        distance = .70 * (candidate['radius'] + cfg['env']['link_radius'])
        side = rng.choice([-1, 1])
        for link in rng.permutation(2):
            for time, fraction in ((.5, .5), (.35, .7), (.65, .7)):
                arm = points(start + time * (goal_q - start))
                direction = arm[link + 1] - arm[link]
                normal = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
                center = arm[link] + fraction * direction
                centers.extend([center + side * distance * normal, center - side * distance * normal])
    else:
        middle = start + rng.uniform(.25, .75) * (goal_q - start)
        arm = points(middle)
        link = rng.integers(2)
        direction = arm[link + 1] - arm[link]
        normal = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
        center = arm[link] + rng.uniform(.25, .90) * direction
        center += rng.uniform(-.25, .25) * candidate['radius'] * normal
        centers.append(center)
    bounds = cfg.get('sampling', {}).get('obstacle_center_radius', DEFAULT_SAMPLING['obstacle_center_radius'])
    for center in centers:
        if not bounds[0] <= np.linalg.norm(center) <= bounds[1]:
            continue
        candidate['obstacle'] = center.tolist()
        if all(clearance(pose, candidate, cfg['env']['link_radius']).min() > cfg['env']['margin']
               for pose in (start, goal_q)):
            return candidate
    return None
