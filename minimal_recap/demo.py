"""Record a fresh expert episode and export its real MuJoCo state replay."""
import argparse
import datetime
from pathlib import Path
import yaml
from .config import load, seed_all, code_version
from .data import collect_episode, atomic_json
from .tasks import make_env, make_expert
from .replay import export_video


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--seed',type=int,default=10000)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    c=load(args.config)
    seed_all(c['seeds']['training'])
    root=args.output or Path('runs')/('expert_video_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    if root.exists() and any(root.iterdir()):
        raise ValueError('Choose an empty output directory to preserve previous recordings')
    root.mkdir(parents=True,exist_ok=True)
    (root/'resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False),encoding='utf-8')
    atomic_json(root/'manifest.json',{'code':code_version(),'policy':'closed-loop expert','seed':args.seed})
    env=make_env(c)
    try:
        episode=collect_episode(env,make_expert(env),args.seed,
            {'episode_id':'expert_demo','source':'expert','checkpoint':None,'control_hz':c['environment']['control_hz'],
             'data_version':c['dataset']['version']},root/'episode.npz')
    finally:
        env.close()
    atomic_json(root/'result.json',episode['metadata'])
    print(episode['metadata'],flush=True)
    robot=c['environment']['robot']
    print(export_video(c,episode,root/'rollout.mp4',title=f'{robot} | closed-loop expert | physical MuJoCo rollout'),flush=True)


if __name__=='__main__':
    main()
