"""Sequential matched runs across training seeds; no hidden parallelism."""
import argparse
from pathlib import Path
import subprocess
import sys
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minimal_recap.config import load

p = argparse.ArgumentParser()
p.add_argument('--config', default='configs/experiments/recap.yaml')
p.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
p.add_argument('--stages', nargs='+')
a = p.parse_args()
folder = Path('runs/seed_configs')
folder.mkdir(parents=True, exist_ok=True)
for seed in a.seeds:
    c = load(a.config)
    c['seeds']['training'] = seed
    c['run']['name'] += f'_seed{seed}'
    path = folder/f'{c["run"]["name"]}.yaml'
    if path.exists() and yaml.safe_load(path.read_text()) != c:
        raise RuntimeError('Seed config already exists with different content')
    path.write_text(yaml.safe_dump(c, sort_keys=False))
    cmd = [sys.executable, '-m', 'minimal_recap.run', '--config', str(path)]
    if a.stages:
        cmd += ['--stages', *a.stages]
    subprocess.run(cmd, check=True)
