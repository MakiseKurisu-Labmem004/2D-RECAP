"""Sample scene geometry independently of the physics simulator."""

import numpy as np


DEFAULT_SAMPLING = {
    'goal_radius': [.40, .70],
    'inner_goal_radius': [.20, .35],
    'outer_goal_radius': [.75, .90],
    'obstacle_center_radius': [.20, .80],
}


def sample_annulus(rng, bounds):
    if len(bounds)!=2 or not np.isfinite(bounds).all() or not 0<=bounds[0]<bounds[1]:
        raise ValueError('Annulus bounds must contain two finite radii with 0 <= min < max')
    angle = rng.uniform(-np.pi, np.pi)
    radius = np.sqrt(rng.uniform(bounds[0]**2, bounds[1]**2))
    return radius * np.array([np.cos(angle), np.sin(angle)])


def sample_scene(rng, split, cfg, seed, attempt):
    q0 = rng.uniform([-2., -.8], [2., 2.4])
    legacy = split in ('goal_ood', 'composition_ood') or any(
        name in cfg.get('data', {}) for name in ('goal_ood', 'composition_ood'))
    if legacy:
        angle = rng.uniform(-1.3, -.15) if split in ('goal_ood', 'composition_ood') else rng.uniform(.15, 1.3)
        radial = rng.uniform(.45, .85)
        goal = radial * np.array([np.cos(angle), np.sin(angle)])
        obstacle = rng.uniform([.15, -.85], [.65, .85])
    else:
        sampling = {**DEFAULT_SAMPLING, **cfg.get('sampling', {})}
        radius_key = 'goal_radius'
        if split in ('inner_goal_ood', 'inner_composition_ood'):
            radius_key = 'inner_goal_radius'
        elif split in ('outer_goal_ood', 'outer_composition_ood'):
            radius_key = 'outer_goal_radius'
        goal = sample_annulus(rng, sampling[radius_key])
        obstacle = sample_annulus(rng, sampling['obstacle_center_radius'])
    large_obstacle = split in ('size_ood', 'inner_composition_ood', 'outer_composition_ood', 'composition_ood')
    radius = rng.uniform(.10, .13) if large_obstacle else rng.uniform(.05, .09)
    return dict(q0=q0.tolist(), goal=goal.tolist(), obstacle=obstacle.tolist(),
                radius=float(radius), scene_id=f'{split}-{seed}-{attempt}', split=split)
