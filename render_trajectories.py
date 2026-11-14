"""Export saved RECAP trajectories to top-down MP4 videos without OpenGL."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory
from tqdm import tqdm


def render_video(path, scene, env, output, show_progress=True):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FFMpegWriter
    from matplotlib.patches import Circle

    from recap.geometry import points

    with np.load(path, allow_pickle=False) as episode:
        qpos = episode["qpos"]
        if qpos.ndim != 2 or qpos.shape[1] != 2 or not len(qpos):
            raise ValueError(f"{path}: qpos must have shape (steps, 2) with steps > 0")
        if not np.isfinite(qpos).all():
            raise ValueError(f"{path}: qpos contains non-finite values")
        # Saved qpos is the state after each control step. Include the initial pose.
        poses = np.vstack([scene["q0"], qpos])
        result = "success" if episode["success"][-1] else str(episode["failure_reason"][-1])

    joints = np.array([points(pose) for pose in poses])
    tip = joints[:, -1]
    dt = env["timestep"] * env["substeps"]
    fps = 1 / dt
    fig, ax = plt.subplots(figsize=(6.4, 6.4), layout="constrained")
    try:
        ax.set(xlim=(-1.08, 1.08), ylim=(-1.08, 1.08), aspect="equal",
               xlabel="x (m)", ylabel="y (m)")
        ax.grid(alpha=0.2)
        ax.add_patch(Circle(scene["obstacle"], scene["radius"],
                            color="tab:red", alpha=0.6, label="Obstacle"))
        ax.add_patch(Circle(scene["goal"], env["success_distance"],
                            fill=False, edgecolor="tab:green", linewidth=2,
                            label="Goal region"))
        ax.plot(*scene["goal"], "+", color="tab:green")
        trail, = ax.plot([], [], "--", color="tab:blue", alpha=0.55,
                         label="End-effector path")
        arm, = ax.plot([], [], color="tab:blue", solid_capstyle="round",
                       solid_joinstyle="round", label="Arm")
        ax.legend(loc="lower left")
        title = ax.set_title(
            f"{scene['scene_id']} | {path.stem}\n"
            f"r = {scene['radius'] * 100:.1f} cm | t = 0.00 s | replay",
            fontsize=9,
        )
        # Match the rendered capsule width to the configured physical radius.
        fig.canvas.draw()
        arm.set_linewidth(2 * env["link_radius"] / 2.16 * ax.bbox.width * 72 / fig.dpi)
        # Reuse the layout calculated above instead of solving it for every frame.
        fig.set_layout_engine("none")
        writer = FFMpegWriter(
            fps=fps, codec="libx264",
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart", "-threads", "1"],
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        # Publish only complete videos, including when replacing an existing export.
        with TemporaryDirectory(dir=output.parent) as temporary:
            video = Path(temporary) / output.name
            with (
                tqdm(total=len(joints)+round(fps),desc=f'Frames {path.stem}',
                     unit='frame',leave=False,dynamic_ncols=True,
                     disable=not show_progress) as progress,
                writer.saving(fig, str(video), dpi=100),
            ):
                for step, xy in enumerate(joints):
                    arm.set_data(xy[:, 0], xy[:, 1])
                    trail.set_data(tip[:step + 1, 0], tip[:step + 1, 1])
                    status = result if step == len(joints) - 1 else "replay"
                    title.set_text(
                        f"{scene['scene_id']} | {path.stem}\n"
                        f"r = {scene['radius'] * 100:.1f} cm | "
                        f"t = {step * dt:.2f} s | {status}"
                    )
                    writer.grab_frame()
                    progress.update(1)
                # Keep the final pose and outcome visible for one second.
                for _ in range(round(fps)):
                    writer.grab_frame()
                    progress.update(1)
            video.replace(output)
    finally:
        plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Run directory, e.g. runs/seed42")
    parser.add_argument(
        "--split",
        choices=["train", "validation", "id", "goal_ood", "size_ood", "composition_ood"],
        help="Export only this split. Default: all splits.",
    )
    parser.add_argument(
        "--source", default="expert",
        choices=["expert", "rollout", "bc", "sft", "recap", "annotated", "all"],
        help="Trajectory source. Default: expert. Use all for every source.",
    )
    parser.add_argument("--limit", type=int, help="Maximum number of matching files to process.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing MP4 exports.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel video export processes. Default: 1.")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    if args.workers < 1:
        parser.error("--workers must be a positive integer")

    run = args.run.resolve()
    for filename in ("scenes.json", "config.json"):
        if not (run / filename).is_file():
            parser.error(f"Missing {run / filename}. Finish recap generate first.")
    data = run / "data"
    split = args.split or "*"
    source = "*" if args.source == "all" else args.source
    paths = sorted(data.glob(f"{split}/{source}/*.npz"))[:args.limit]
    if not paths:
        parser.error(f"No matching trajectories in {data} (split={split}, source={source})")

    import numpy as np
    from matplotlib.animation import writers

    if not writers.is_available("ffmpeg"):
        parser.error("FFmpeg is unavailable. Install ffmpeg and add it to PATH.")
    groups = json.loads((run / "scenes.json").read_text(encoding="utf-8"))
    scenes = {scene["scene_id"]: scene for group in groups.values() for scene in group}
    env = json.loads((run / "config.json").read_text(encoding="utf-8"))["env"]
    dt = env["timestep"] * env["substeps"]
    if not math.isfinite(dt) or dt <= 0:
        parser.error("env.timestep * env.substeps must be finite and positive")

    with tqdm(total=len(paths),desc="Export videos",unit="video",dynamic_ncols=True) as progress:
        jobs = []
        for path in paths:
            output = run / "videos" / path.relative_to(data).with_suffix(".mp4")
            if output.exists() and not args.overwrite:
                tqdm.write(f"Skip existing {output}")
                progress.update(1)
                continue
            with np.load(path, allow_pickle=False) as episode:
                scene_id = episode["scene_id"].item()
            if scene_id not in scenes:
                parser.error(f"{path}: scene {scene_id!r} is missing from scenes.json")
            jobs.append((path, scenes[scene_id], env, output))

        if args.workers == 1:
            for job in jobs:
                tqdm.write(str(render_video(*job)))
                progress.update(1)
        elif jobs:
            # Spawn isolates Matplotlib state and works on both Windows and Linux.
            with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs)),
                                     mp_context=get_context("spawn")) as executor:
                futures = {executor.submit(render_video, *job, show_progress=False): job[0]
                           for job in jobs}
                try:
                    for future in as_completed(futures):
                        try:
                            output = future.result()
                        except Exception as exc:
                            raise RuntimeError(f"Failed to export {futures[future]}: {exc}") from exc
                        tqdm.write(str(output))
                        progress.update(1)
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise


if __name__ == "__main__":
    main()
