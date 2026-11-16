"""Replay saved physical states in a viewer, GIF, or H.264 MP4."""
import argparse
from pathlib import Path
import time
import numpy as np
import yaml
import mujoco
from .data import read_episode
from .env import PlanarGraspEnv


def export_video(config, episode, output, stride=1, title=None):
    from PIL import Image, ImageDraw
    if stride < 1:
        raise ValueError('stride must be positive')
    output = Path(output)
    if output.suffix.lower() not in ['.gif', '.mp4']:
        raise ValueError('Output must end in .gif or .mp4')
    output.parent.mkdir(parents=True, exist_ok=True)
    env = PlanarGraspEnv(config)
    env.restore(episode['states'][0])
    fps = config['environment']['control_hz']/stride
    indices = list(range(0, len(episode['states']), stride))
    if indices[-1] != len(episode['states'])-1:
        indices.append(len(episode['states'])-1)
    writer, images = None, []
    try:
        if output.suffix.lower() == '.mp4':
            import imageio.v2 as imageio
            writer = imageio.get_writer(str(output), fps=fps, codec='libx264', quality=8, pixelformat='yuv420p')
        for i in indices:
            state = episode['states'][i]
            # Same model throughout an episode: preserve the rendering context.
            mujoco.mj_setState(env.model, env.data, np.asarray(state['physics']), mujoco.mjtState.mjSTATE_INTEGRATION)
            mujoco.mj_forward(env.model, env.data)
            frame = Image.fromarray(env.render())
            if title:
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, frame.width, 53), fill=(18, 22, 28))
                draw.text((12, 8), title, fill='white')
                phase = episode['controller_states'][i]['stage'] if episode.get('controller_states') else 'network policy'
                result = episode['metadata'].get('reason', '') if i == len(episode['states'])-1 else phase
                draw.text((12, 29), f't={i/config["environment"]["control_hz"]:.2f}s  step={i}  {result}', fill=(210, 220, 230))
            if writer:
                writer.append_data(np.asarray(frame))
            else:
                images.append(frame)
        if images:
            images[0].save(output, save_all=True, append_images=images[1:], loop=0, duration=round(1000/fps))
    finally:
        if writer:
            writer.close()
        env.close()
    return {'path': str(output.resolve()), 'frames': len(indices), 'fps': fps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='MP4 or GIF; omitted opens native MuJoCo viewer')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--title')
    args = parser.parse_args()
    if args.stride < 1:
        parser.error('--stride must be positive')
    c = yaml.safe_load((args.run/'resolved.yaml').read_text(encoding='utf-8'))
    ep = read_episode(args.episode)
    if args.output:
        print(export_video(c, ep, args.output, args.stride, args.title))
    else:
        from mujoco import viewer as mj_viewer
        env = PlanarGraspEnv(c)
        env.restore(ep['states'][0])
        try:
            with mj_viewer.launch_passive(env.model, env.data) as viewer:
                for state in ep['states'][::args.stride]:
                    if not viewer.is_running():
                        break
                    mujoco.mj_setState(env.model, env.data, np.asarray(state['physics']), mujoco.mjtState.mjSTATE_INTEGRATION)
                    mujoco.mj_forward(env.model, env.data)
                    viewer.sync()
                    time.sleep(args.stride/c['environment']['control_hz'])
        finally:
            env.close()


if __name__ == '__main__':
    main()
