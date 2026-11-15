"""Checkpoint dataset generation and recover expert-only interrupted runs."""

from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import SimpleNamespace

import numpy as np
from tqdm import tqdm

from .datasets import DATASET_SPLITS, LEGACY_DATASET_SPLITS
from .difficulty import difficulty_quotas
from .geometry import points
from .planning import ScenarioGenerator, ik, local_medium_detour, plan, safe_edge
from .observations import without_velocity, convert_episode
from .controllers import make_expert


def write_json(path, value):
    path = Path(path)
    temporary = None
    try:
        with NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def generation_lock(root):
    with (root / '.generation.lock').open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            if stream.tell() == 0:
                stream.write(b'0')
                stream.flush()
            stream.seek(0)
        else:
            import fcntl
        try:
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f'Generation is already running for {root}') from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def read_json(path, default):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def matches_expert(path, episode, start, limit, cfg=None):
    state = 1
    before = np.vstack([start, episode['qpos'][:-1]])
    controller = str(episode.get('expert_controller', 'waypoint'))
    if controller != 'waypoint':
        if cfg is None:
            raise ValueError('Smooth expert verification requires its saved configuration')
        expert = make_expert(path, {**cfg, 'planner': {**cfg['planner'], 'expert_controller': controller}})
        for pose, action in zip(before, episode['action']):
            env = SimpleNamespace(data=SimpleNamespace(qpos=pose))
            if not np.allclose(expert(None, env), action, atol=3e-6, rtol=0):
                return False
        return True
    for pose, action in zip(before, episode['action']):
        if np.max(np.abs(path[state] - pose)) < .025 and state < len(path) - 1:
            state += 1
        if not np.allclose(np.clip(path[state] - pose, -limit, limit), action, atol=3e-6, rtol=0):
            return False
    return True


def path_from_controls(episode, scene, start, cfg):
    limit = cfg['env']['action_limit']
    targets = []
    unsaturated = (np.abs(episode['action']) < limit - 1e-9).all(axis=1)
    for target in episode['applied_command'][unsaturated]:
        if not targets or not np.allclose(target, targets[-1], atol=1e-9, rtol=0):
            targets.append(target)
    if not targets or not np.allclose(points(targets[-1])[-1], scene['goal'], atol=2e-6, rtol=0):
        return None
    if not safe_edge(start, targets[0], scene, cfg):
        grid = np.linspace(-cfg['env']['joint_limit'], cfg['env']['joint_limit'], cfg['planner']['resolution'])
        nearest = grid[np.abs(grid[:, None] - start).argmin(axis=0)]
        targets.insert(0, nearest)
    path = np.vstack([start, targets])
    if not all(safe_edge(a, b, scene, cfg) for a, b in zip(path[:-1], path[1:])):
        return None
    return path if matches_expert(path, episode, start, limit) else None


def recover_expert_path(episode, scene, cfg):
    start = np.asarray(scene['q0'])
    if 'expert_path' in episode:
        path = np.asarray(episode['expert_path'])
        if (not np.allclose(path[0], start, atol=2e-6, rtol=0)
                or not all(safe_edge(a, b, scene, cfg) for a, b in zip(path[:-1], path[1:]))
                or not matches_expert(path, episode, start, cfg['env']['action_limit'], cfg)):
            raise ValueError('Saved expert path does not match the trajectory')
        return path
    path = path_from_controls(episode, scene, start, cfg)
    if path is not None:
        return path
    goals = ik(np.array(scene['goal']), cfg['env']['joint_limit'])
    goals.sort(key=lambda goal: np.linalg.norm(goal - episode['qpos'][-1]))
    for goal in goals:
        direct = safe_edge(start, goal, scene, cfg)
        candidate = np.array([start, goal]) if direct else local_medium_detour(start, goal, scene, cfg)
        if candidate is None or not matches_expert(candidate, episode, start, cfg['env']['action_limit']):
            candidate = plan(start, goal, scene, cfg)
        if candidate is not None and matches_expert(candidate, episode, start, cfg['env']['action_limit']):
            return candidate
    raise ValueError(f"Cannot verify the expert path for {scene['scene_id']}")


def recover_scene(episode, split, cfg):
    if 'scene_json' in episode:
        return json.loads(episode['scene_json'].item())
    if not episode['success'][-1] or float(episode['exploration']) != 0:
        raise ValueError('Recovery requires successful, noise-free expert trajectories')
    obs = without_velocity(episode['observation'][0])
    start = episode['applied_command'][0] - episode['action'][0]
    if not np.allclose(np.r_[np.sin(start), np.cos(start)], obs[:4], atol=2e-6, rtol=0):
        raise ValueError('Cannot recover the initial joint angles from saved commands')
    scene = dict(scene_id=episode['scene_id'].item(), split=split, q0=start.tolist(),
                 goal=obs[4:6].tolist(), obstacle=obs[6:8].tolist(), radius=float(obs[8]))
    path = recover_expert_path(episode, scene, cfg)
    direct = safe_edge(start, path[-1], scene, cfg)
    length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    ratio = length / max(np.linalg.norm(path[-1] - start), 1e-8)
    gap = float(episode['clearance'].min())
    difficulty = 'easy' if direct else 'hard' if (
        ratio >= cfg['planner']['hard_ratio'] and gap < cfg['planner']['hard_clearance']) else 'medium'
    tips = np.array([points(q)[-1] for q in episode['qpos']])
    scene['metrics'] = dict(steps=len(episode['reward']), joint_path_length=length,
                            ee_path_length=float(np.linalg.norm(np.diff(tips, axis=0), axis=1).sum()),
                            detour_ratio=float(ratio), min_clearance=gap, difficulty=difficulty)
    scene['recovered_from_expert'] = True
    return scene


def store_expert(root, split, scene, episode):
    folder = root / 'data' / split / 'expert'
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / f"{episode['episode_id'].item()}.npz"
    temporary = None
    try:
        with NamedTemporaryFile(dir=folder, delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **{**convert_episode(episode), 'scene_json': np.array(json.dumps(scene))})
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def generate_datasets(root, cfg, selected=None, resume=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with generation_lock(root):
        saved_config = read_json(root / 'config.json', None)
        if saved_config is not None and saved_config != cfg:
            raise ValueError('Use the saved run configuration when resuming generation')
        if saved_config is None:
            write_json(root / 'config.json', cfg)
        return generate_locked(root, cfg, selected, resume)


def generate_locked(root, cfg, selected, resume):
    splits = LEGACY_DATASET_SPLITS if any(name in cfg['data'] for name in ('goal_ood', 'composition_ood')) else DATASET_SPLITS
    selected = set(splits if selected is None else selected)
    if not selected.issubset(splits):
        raise ValueError(f'Dataset selection is incompatible with this configuration: {selected - set(splits)}')
    scenes = read_json(root / 'scenes.json', {})
    existing = any(next((root / 'data' / split / 'expert').glob('*.npz'), None) for split in splits)
    if not resume and (scenes or existing):
        raise FileExistsError('Existing datasets found. Use generate --resume to complete missing datasets.')
    baseline = read_json(root / 'generation.json', {})
    summaries = read_json(root / 'difficulty.json', {})
    generator = ScenarioGenerator(cfg)
    pending = {}
    for split in splits:
        paths = sorted((root / 'data' / split / 'expert').glob('*.npz'))
        known = {scene['scene_id']: scene for scene in scenes.get(split, [])}
        pairs = []
        with tqdm(paths, desc=f'Check existing {split}', unit='episode', leave=False, dynamic_ncols=True) as progress:
            for path in progress:
                with np.load(path, allow_pickle=False) as archive:
                    episode = dict(archive)
                scene = known.get(episode['scene_id'].item())
                if scene is None:
                    try:
                        scene = recover_scene(episode, split, cfg)
                    except ValueError as error:
                        raise ValueError(f'{path}: {error}') from error
                if scene['scene_id'] != episode['scene_id'].item() or scene.get('split', split) != split:
                    raise ValueError(f'{path}: scene metadata and episode identity disagree')
                pairs.append((scene, episode))
        if pairs and all('seed' in episode for scene, episode in pairs):
            pairs.sort(key=lambda pair: int(pair[1]['seed']))
        if len(pairs) > cfg['data'][split]:
            raise ValueError(f'{split}: more expert episodes than configured scenes')
        if known and set(known) != {scene['scene_id'] for scene, episode in pairs}:
            raise ValueError(f'{split}: manifest and expert episode files disagree')
        if len({scene['scene_id'] for scene, episode in pairs}) != len(pairs):
            raise ValueError(f'{split}: duplicate scene IDs')
        if len(pairs) == cfg['data'][split]:
            actual = Counter(scene['metrics']['difficulty'] for scene, episode in pairs)
            if cfg.get('difficulty_ratio') is not None:
                target = difficulty_quotas(len(pairs), cfg['difficulty_ratio'])
                if any(actual[name] != value for name, value in target.items()):
                    raise ValueError(f'{split}: recovered difficulty counts disagree with configured quotas')
            scenes[split] = [scene for scene, episode in pairs]
            if split not in summaries:
                summaries[split] = dict(target=difficulty_quotas(len(pairs), cfg['difficulty_ratio'])
                                        if cfg.get('difficulty_ratio') is not None else None,
                                        accepted={name: actual[name] for name in ('easy', 'medium', 'hard')},
                                        attempts=None, recovered=True)
            write_json(root / 'scenes.json', scenes)
            write_json(root / 'difficulty.json', summaries)
            tqdm.write(f'Skip complete {split}: {len(pairs)} scenes')
        else:
            pending[split] = pairs
    try:
        for index, split in enumerate(splits):
            if split not in selected or split not in pending:
                continue
            pairs = generator.generate(split, cfg['data'][split], cfg['seed'] + index * 1000000,
                                       initial_pairs=pending[split])
            old_ids = {episode['episode_id'].item() for scene, episode in pending[split]}
            for scene, episode in pairs:
                if episode['episode_id'].item() not in old_ids:
                    store_expert(root, split, scene, episode)
            scenes[split] = [scene for scene, episode in pairs]
            write_json(root / 'scenes.json', scenes)
            tqdm.write(f'{split} {len(pairs)}')
    finally:
        totals = baseline.copy()
        for name, value in generator.rejections.items():
            totals[name] = totals.get(name, 0) + value
        summaries.update(generator.difficulty_summary)
        write_json(root / 'generation.json', totals)
        write_json(root / 'difficulty.json', summaries)
