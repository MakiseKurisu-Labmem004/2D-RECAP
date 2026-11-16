"""Export viewer-only Value / n-step Advantage caches from existing round critics."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def export(run, output):
    import numpy as np
    import torch
    from recap.models import Model, advantage, returns
    from recap.rewards import REWARD_VERSION, return_scale, relabel_rewards

    torch.set_num_threads(1)
    count = 0
    for root in sorted((run / 'improvement').glob('*/*/round-*')):
        checkpoint = root / 'value.pt'
        if not checkpoint.exists():
            continue
        cfg = json.loads((root / 'config.json').read_text(encoding='utf-8'))
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model_cfg = ck.get('config', cfg)
        if model_cfg['env'] != cfg['env']:
            raise ValueError(f'{checkpoint}: environment does not match round config')
        model = Model('value', model_cfg['train'], model_cfg['env']['action_limit'])
        model.load_state_dict(ck['model'])
        model.eval()
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        current_rewards = model_cfg['train'].get('reward_version') == REWARD_VERSION
        scale = return_scale(cfg['env']['horizon']) if current_rewards else cfg['env']['horizon'] + 149
        n = cfg['train']['n_step']
        diagnostics_path = root / 'diagnostics.json'
        diagnostics = json.loads(diagnostics_path.read_text()) if diagnostics_path.exists() else {}
        threshold = diagnostics.get('threshold', cfg['train']['threshold'])
        paths = sorted((root / 'data' / 'adaptation').glob('*/*.npz'))
        for path in paths:
            with np.load(path, allow_pickle=False) as archive:
                obs, next_obs, reward = archive['observation'], archive['next_observation'], archive['reward']
                offset = int(archive['control_step_offset']) if 'control_step_offset' in archive else 0
                if current_rewards:
                    episode = dict(archive)
                    relabel_rewards(episode, cfg['env']['horizon'])
                    reward = episode['reward']
            with torch.inference_mode():
                values = model.value(torch.as_tensor(obs, dtype=torch.float32)).numpy()
                next_values = model.value(torch.as_tensor(next_obs, dtype=torch.float32)).numpy()
            adv = advantage(reward, values, next_values, n, scale)
            payload = dict(value_prediction=values.tolist(), mc_return=returns(reward, scale).tolist(),
                           td_advantage=adv.tolist(), advantage_indicator=(adv > threshold).tolist(),
                           value_alignment='observation_before_action', n_step=n, return_scale=scale,
                           advantage_threshold=threshold, control_step_offset=offset,
                           within_task_horizon=(np.arange(len(reward)) + offset < cfg['env']['horizon']).tolist(),
                           value_checkpoint=checkpoint.relative_to(run).as_posix(), value_checkpoint_sha256=digest,
                           value_checkpoint_step=ck.get('global_step'),
                           trajectory_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           annotation_origin='viewer_recomputed_from_saved_checkpoint',
                           annotation_note='TD threshold only; excludes forced high-condition labels for experts/interventions.')
            destination = output / path.relative_to(run).with_suffix('.json')
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix('.tmp')
            temporary.write_text(json.dumps(payload, allow_nan=False, separators=(',', ':')), encoding='utf-8')
            temporary.replace(destination)
            count += 1
        print(f'{root.relative_to(run)}: {len(paths)} episodes', flush=True)
    print(f'Exported {count} viewer annotations to {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path(__file__).resolve().parent.parent / 'runs' / 'seed42')
    args = parser.parse_args()
    run = args.run.resolve()
    export(run, Path(__file__).resolve().parent / '.cache' / 'value' / run.name)
