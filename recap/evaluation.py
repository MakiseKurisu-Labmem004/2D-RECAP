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
    return result


def difficulty_metrics(episodes, scenes):
    labels = {scene['scene_id']: scene.get('metrics', {}).get('difficulty', 'unknown') for scene in scenes}
    groups = {}
    for ep in episodes:
        groups.setdefault(labels[str(ep['scene_id'])], []).append(ep)
    return {name: episode_metrics(group) for name, group in groups.items()}
