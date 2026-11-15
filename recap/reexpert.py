"""Replay new experts on fixed scenes, preserving the original difficulty groups."""

import copy
import json
from pathlib import Path
import shutil
from tempfile import mkdtemp

import numpy as np
from tqdm import tqdm

from .controllers import make_expert
from .env import ObstacleReach2D
from .generation import recover_expert_path, store_expert, write_json
from .planning import rollout, completed_goal_path


def motion_metrics(episode, scene, cfg):
    dt = cfg['env']['timestep'] * cfg['env']['substeps']
    poses = np.vstack((scene['q0'], episode['qpos']))
    velocity = np.diff(poses, axis=0) / dt
    acceleration = np.diff(velocity, axis=0) / dt
    jerk = np.diff(acceleration, axis=0) / dt
    def peak(values):
        return float(np.max(np.linalg.norm(values, axis=1))) if len(values) else 0.
    result = dict(steps=len(episode['action']), peak_joint_speed=peak(velocity),
                peak_joint_acceleration=peak(acceleration), peak_joint_jerk=peak(jerk),
                peak_action_jump=peak(np.diff(episode['action'], axis=0)),
                min_clearance=float(episode['clearance'].min()))
    if 'substep_qpos' in episode:
        physics = np.vstack((scene['q0'], np.concatenate([q[:int(n)] for q, n in
                             zip(episode['substep_qpos'], episode['executed_substeps'])])))
        result['peak_physics_joint_jerk'] = peak(np.diff(physics, n=3, axis=0) / cfg['env']['timestep']**3)
    return result


def rebuild_experts(source, destination, settings):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Expert replay requires separate source and destination directories')
    if destination.exists():
        raise FileExistsError(f'Destination already exists: {destination}')
    original = json.loads((source / 'config.json').read_text())
    cfg = copy.deepcopy(original)
    cfg['planner'].update({k: v for k, v in settings['planner'].items() if k.startswith('expert_')})
    cfg['train'].update(settings['train'])
    scenes = json.loads((source / 'scenes.json').read_text())
    paths = sorted((source / 'data').glob('*/expert/*.npz'))
    lookup = {s['scene_id']: s for split in scenes.values() for s in split}
    if len(paths) != len(lookup):
        raise ValueError('Expert replay requires one expert file per scene in scenes.json')
    destination.parent.mkdir(parents=True, exist_ok=True)
    prefix = f'.{destination.name}.experts-'
    temporary = Path(mkdtemp(prefix=prefix, dir=destination.parent)).resolve()
    env = ObstacleReach2D(cfg['env'])
    report = []
    seen = set()
    try:
        with tqdm(paths, desc='Replay smooth experts', unit='episode', dynamic_ncols=True) as progress:
            for path in progress:
                with np.load(path, allow_pickle=False) as archive:
                    previous = dict(archive)
                scene_id = str(previous['scene_id'])
                if scene_id in seen: raise ValueError(f'Duplicate expert for {scene_id}')
                seen.add(scene_id)
                scene = lookup[scene_id]
                route = recover_expert_path(previous, scene, original)
                route = completed_goal_path(route, previous, scene, cfg)
                expert = make_expert(route, cfg, scene)
                current = rollout(env, scene, expert, int(previous['seed']), 'expert', limit=cfg['planner']['expert_limit'])
                if not current['success'][-1]:
                    raise RuntimeError(f'{scene_id}: smooth expert failed ({current["failure_reason"][-1]}); original data unchanged')
                current.update(expert_path=route, expert_controller=np.array(expert.kind), episode_id=previous['episode_id'])
                metrics = motion_metrics(current, scene, cfg)
                scene['expert_replay_metrics'] = dict(controller=expert.kind, **metrics)
                store_expert(temporary, scene['split'], scene, current)
                report.append(dict(scene_id=scene_id, original=motion_metrics(previous, scene, original), replay=metrics))
        write_json(temporary / 'config.json', cfg)
        write_json(temporary / 'scenes.json', scenes)
        write_json(temporary / 'expert_replay.json', dict(source=str(source), episodes=report,
                   note='Scene geometry and original difficulty groups are fixed; new execution metrics are stored separately.'))
        temporary.rename(destination)
    finally:
        env.close()
        if temporary.exists():
            if temporary.parent != destination.parent or not temporary.name.startswith(prefix):
                raise RuntimeError('Unexpected expert replay staging directory')
            shutil.rmtree(temporary)
    return destination
