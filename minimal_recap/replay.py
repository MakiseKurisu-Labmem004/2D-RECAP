import argparse
from pathlib import Path
import time
import numpy as np
import yaml
from .data import read_episode
from .env import PlanarGraspEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='Write an animated GIF; otherwise use native MuJoCo viewer')
    parser.add_argument('--stride', type=int, default=1)
    args = parser.parse_args()
    c = yaml.safe_load((args.run/'resolved.yaml').read_text(encoding='utf-8'))
    ep = read_episode(args.episode)
    env = PlanarGraspEnv(c)
    if args.output:
        from PIL import Image
        frames = []
        for state in ep['states'][::args.stride]:
            env.restore(state)
            frames.append(Image.fromarray(env.render()))
        frames[0].save(args.output, save_all=True, append_images=frames[1:], loop=0,
                       duration=round(1000*args.stride/c['environment']['control_hz']))
    else:
        import mujoco.viewer
        env.restore(ep['states'][0])
        # Model geometry is constant within one episode. Restore integration state in place.
        import mujoco
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            for state in ep['states'][::args.stride]:
                if not viewer.is_running():
                    break
                mujoco.mj_setState(env.model, env.data, np.asarray(state['physics']), mujoco.mjtState.mjSTATE_INTEGRATION)
                mujoco.mj_forward(env.model, env.data)
                viewer.sync()
                time.sleep(args.stride/c['environment']['control_hz'])
    env.close()


if __name__ == '__main__':
    main()
