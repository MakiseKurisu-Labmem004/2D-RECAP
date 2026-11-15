"""Success and failure metrics shared by policy evaluations."""

import numpy as np


def episode_metrics(episodes):
    n = len(episodes)
    if not n:
        raise ValueError('Evaluation requires at least one episode')
    successes = [bool(ep['success'][-1]) for ep in episodes]
    rate = float(np.mean(successes))
    center = (rate + 1.96**2 / (2 * n)) / (1 + 1.96**2 / n)
    half = 1.96 * np.sqrt(rate * (1 - rate) / n + 1.96**2 / (4 * n * n)) / (1 + 1.96**2 / n)
    result = dict(
        episodes=n, success_count=sum(successes), success_rate=rate,
        success_ci95=[float(center - half), float(center + half)],
        collision_rate=float(np.mean(['collision' in str(ep['failure_reason'][-1]) for ep in episodes])),
        timeout_rate=float(np.mean([ep['truncated'][-1] for ep in episodes])),
        completion_steps=float(np.mean([len(ep['reward']) for ep in episodes if ep['success'][-1]]))
        if any(successes) else None,
        scene_ids=[str(ep['scene_id']) for ep in episodes], successes=successes,
    )
    if all('value_prediction' in ep and 'mc_return' in ep for ep in episodes):
        result['value_mae'] = float(np.mean(np.abs(np.concatenate([
            ep['value_prediction'] - ep['mc_return'] for ep in episodes]))))
    if all('distance' in ep for ep in episodes):
        result['mean_final_distance'] = float(np.mean([ep['distance'][-1] for ep in episodes]))
        result['mean_min_distance'] = float(np.mean([np.min(ep['distance']) for ep in episodes]))
    if all('control_dt' in ep for ep in episodes):
        times = [len(ep['reward'])*float(ep['control_dt']) for ep in episodes if ep['success'][-1]]
        result['completion_seconds'] = float(np.mean(times)) if times else None
    if all(all(key in ep for key in ('distance','strict_success_distance','goal_hold_required')) for ep in episodes):
        strict = []
        for ep in episodes:
            held, achieved = 0, False
            for distance in ep['distance']:
                held = held+1 if distance < float(ep['strict_success_distance']) else 0
                achieved = achieved or held >= int(ep['goal_hold_required'])
            strict.append(bool(ep['success'][-1]) and achieved)
        result['strict_success_count'] = sum(strict)
        result['strict_success_rate'] = float(np.mean(strict))
        result['tolerance_only_success_count'] = sum(successes)-sum(strict)
    if all(all(key in ep for key in ('distance', 'qpos', 'success_distance')) for ep in episodes):
        stalls = [bool(ep['truncated'][-1]) and len(ep['qpos']) >= 20
                  and float(ep['success_distance']) < ep['distance'][-1] <= 2*float(ep['success_distance'])
                  and np.max(np.linalg.norm(np.diff(ep['qpos'][-20:], axis=0), axis=1)) < 1e-5
                  for ep in episodes]
        result['near_goal_stall_count'] = int(sum(stalls))
        result['near_goal_stall_rate'] = float(np.mean(stalls))
    return result


def difficulty_metrics(episodes, scenes):
    labels = {scene['scene_id']: scene.get('metrics', {}).get('difficulty', 'unknown') for scene in scenes}
    groups = {}
    for ep in episodes:
        groups.setdefault(labels[str(ep['scene_id'])], []).append(ep)
    return {name: episode_metrics(group) for name, group in groups.items()}
