"""Aggregate independent training seeds; never count repeated deterministic tests as seeds."""
import argparse
from pathlib import Path
import json
import numpy as np
import yaml
from .data import atomic_json


def flatten(obj, prefix=''):
    out = {}
    for k, v in obj.items():
        key = f'{prefix}.{k}' if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        elif isinstance(v, (int, float)):
            out[key] = v
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    records, seeds, contracts, configs = [], [], [], []
    for root in a.runs:
        c = yaml.safe_load((root/'resolved.yaml').read_text(encoding='utf-8'))
        configs.append({k: c[k] for k in ['environment', 'observation', 'reward', 'model', 'policy_rollout', 'evaluation', 'bc_training', 'value_training', 'recap']})
        seeds.append(c['seeds']['training'])
        contracts.append(json.loads((root/'contract.json').read_text()))
        records.append(flatten(json.loads((root/'metrics.json').read_text())))
    if len(set(seeds)) != len(seeds):
        raise ValueError('Every run must have an independent training seed')
    if any(x != contracts[0] for x in contracts) or any(x != configs[0] for x in configs):
        raise ValueError('Experiment contracts/configuration differ across seed runs')
    common = set.intersection(*(set(x) for x in records))
    stats = {k: {'mean': float(np.mean([r[k] for r in records])),
                 'sample_std': float(np.std([r[k] for r in records], ddof=1)) if len(records)>1 else None,
                 'training_seeds': len(seeds)} for k in sorted(common)}
    atomic_json(a.output, {'runs': [str(x.resolve()) for x in a.runs], 'seeds': seeds, 'metrics': stats})


if __name__ == '__main__':
    main()
