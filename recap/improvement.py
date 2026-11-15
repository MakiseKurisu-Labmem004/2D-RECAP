"""Paired continual-improvement experiments with shared original training data."""

import argparse
from collections import deque
import copy
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
import yaml
from tqdm import tqdm

from .configuration import check_environment
from .controllers import make_expert
from .datasets import TEST_SPLITS
from .env import ObstacleReach2D
from .evaluation import episode_metrics
from .generation import write_json
from .geometry import points
from .planning import ik, plan, rollout
from .splines import UnsafeSplineError


def partition_scenes(scenes, fraction, seed):
    """A fixed target-domain holdout, never used for collection or fitting."""
    if not scenes: raise ValueError('No practice scenes selected')
    if fraction==0: return list(scenes),[]
    rng = np.random.default_rng(seed)
    adapt, holdout = [], []
    groups = {}
    for scene in scenes:
        key = (scene.get('split', ''), scene.get('metrics', {}).get('difficulty', 'unknown'))
        groups.setdefault(key, []).append(scene)
    for group in groups.values():
        order = rng.permutation(len(group))
        count = min(len(group)-1, max(1, round(len(group)*fraction))) if fraction and len(group)>1 else 0
        holdout.extend(group[i] for i in order[:count])
        adapt.extend(group[i] for i in order[count:])
    if fraction and not holdout and len(adapt)>1:
        holdout.append(adapt.pop())
    if not adapt: raise ValueError('No adaptation scenes remain')
    return adapt, holdout


def select_scene_pool(manifest, splits, scene_ids=None):
    """Scene source labels are metadata, not distinct learning tasks."""
    if 'all' in splits:
        if len(splits)!=1: raise ValueError('Use all alone, or explicitly list scene sources')
        splits=[name for name,scenes in manifest.items() if name!='validation' and scenes]
    if 'validation' in splits: raise ValueError('validation is reserved for monitoring')
    missing=set(splits)-set(manifest)
    if missing: raise ValueError(f'Unknown scene sources: {sorted(missing)}')
    scenes=[]
    for name in splits:
        for original in manifest[name]:
            scene=copy.deepcopy(original)
            scene['practice_source']=name
            scenes.append(scene)
    if scene_ids:
        missing=set(scene_ids)-{s['scene_id'] for s in scenes}
        if missing: raise ValueError(f'Unknown target scene IDs: {sorted(missing)}')
        scenes=[s for s in scenes if s['scene_id'] in scene_ids]
    if not scenes: raise ValueError('No practice scenes selected')
    if len({s['scene_id'] for s in scenes})!=len(scenes): raise ValueError('Duplicate scene IDs in practice pool')
    counts={name:sum(s['practice_source']==name for s in scenes) for name in splits}
    return scenes,{name:count for name,count in counts.items() if count}


def regression_changes(previous, candidate, validation_previous, validation_candidate):
    regressions = [key for key in previous if previous[key] and not candidate[key]]
    validation_regressions = [key for key in validation_previous if validation_previous[key] and not validation_candidate[key]]
    recovered = [key for key in previous if not previous[key] and candidate[key]]
    return dict(recovered=recovered, regressions=regressions,validation_regressions=validation_regressions)


def takeover_episode(scene, failed, cfg, seed, lookback):
    """Replay actual actions, then keep the expert in control until success.

    Selection is retrospective training-time oracle assistance. No target expert
    trajectory files are read; only the known goal and collision geometry are used.
    """
    env = ObstacleReach2D(cfg['env'])
    used = 0
    try:
        cut = max(0,len(failed['action'])-lookback)
        env.reset(seed=int(failed['seed']), options={'scene': scene})
        for action in failed['action'][:cut]:
            env.step(action)
            if env.done: return None, dict(status='replay_terminated', expert_steps=0, replay_steps=cut)
        if not np.allclose(env.obs(), failed['observation'][cut], atol=2e-6, rtol=0):
            raise ValueError('Takeover replay does not reproduce the original failure state')
        start = env.data.qpos.copy()
        goals = sorted(ik(np.asarray(scene['goal']), cfg['env']['joint_limit']), key=lambda q: np.linalg.norm(q-start))
        expert = None
        for goal in goals:
            route = plan(start, goal, scene, cfg)
            if route is None: continue
            try: expert = make_expert(route, cfg, scene)
            except UnsafeSplineError: continue
            break
        if expert is None: return None, dict(status='planning_failed', expert_steps=0, replay_steps=cut)
        recent=deque(maxlen=25)
        class ExpertStalled(RuntimeError): pass
        def control(obs, current):
            nonlocal used
            recent.append(current.data.qpos.copy())
            finished=(hasattr(expert,'trajectory') and expert.time>=expert.trajectory.duration)
            if finished and len(recent)==recent.maxlen and np.max(np.ptp(np.asarray(recent),axis=0))<1e-8:
                if np.linalg.norm(points(current.data.qpos)[-1]-scene['goal'])>=cfg['env']['success_distance']:
                    raise ExpertStalled('Finished reference is stationary outside the goal')
            action=expert(obs,current)
            used+=1
            return action
        try: ep = rollout(env,scene,control,seed,'takeover',reset=False,until_done=True)
        except ExpertStalled:
            return None,dict(status='expert_stalled',expert_steps=used,replay_steps=cut)
        if not ep['success'][-1]:
            return None,dict(status='expert_execution_failed',expert_steps=used,replay_steps=cut,
                             failure_reason=str(ep['failure_reason'][-1]))
        task_success=cut+len(ep['action'])<=cfg['env']['horizon']
        ep.update(intervention_mask=np.ones(len(ep['action']),dtype=bool), action_source=np.array('expert'),
                  parent_episode_id=failed['episode_id'], takeover_step=np.array(cut),
                  replay_steps=np.array(cut), initial_qpos=start, control_step_offset=np.array(cut),
                  task_horizon=np.array(cfg['env']['horizon']),task_success=np.array(task_success),
                  expert_completion_step=np.array(cut+len(ep['action'])),
                  scene_json=np.array(json.dumps(scene)))
        return ep, dict(status='collected', expert_steps=used, replay_steps=cut,
                        policy_suffix_steps=0,success=True,task_success=task_success,takeover_step=cut,
                        expert_completion_step=cut+len(ep['action']))
    finally:
        env.close()


def correction_value_episode(ep,cfg):
    """Keep the original task deadline for Value; extended expert data teaches policy."""
    remaining=cfg['env']['horizon']-int(ep['takeover_step'])
    if remaining<=0: raise ValueError('Takeover must begin before the original task deadline')
    if len(ep['action'])<=remaining: return ep
    result=dict(ep)
    keys=('observation','action','next_observation','reward','terminated','truncated','success',
          'failure_reason','applied_command','step_index','timestamp','qpos','clearance','distance',
          'goal_hold_steps','control_commands','substep_qpos','executed_substeps','intervention_mask')
    for key in keys:
        if key in ep: result[key]=ep[key][:remaining].copy()
    result['terminated'][-1]=False;result['truncated'][-1]=True;result['success'][:]=False
    result['reward'][-1]=-150.
    result['failure_reason']=result['failure_reason'].astype('<U32')
    result['failure_reason'][-1]='timeout'
    result['value_task_prefix']=np.array(True)
    return result


def evaluate_scenes(model, scenes, cfg, seed, description,*,record_root=None,record_source='practice',value=None):
    from .cli import policy,annotate,store
    if not scenes: return dict(metrics=None, successes={})
    env = ObstacleReach2D(cfg['env'])
    try:
        episodes = [rollout(env, scene, policy(model), seed+i, 'regression')
                    for i, scene in enumerate(tqdm(scenes, desc=description, unit='episode', dynamic_ncols=True))]
    finally: env.close()
    if value is not None:
        annotate(episodes,value,cfg)
    value_info=getattr(value,'checkpoint_info',{}) if value is not None else {}
    value_digest=sha256(Path(value_info['path']).read_bytes()).hexdigest() if value_info.get('path') else ''
    policy_info=getattr(model,'checkpoint_info',{})
    policy_digest=sha256(Path(policy_info['path']).read_bytes()).hexdigest() if record_root is not None and policy_info.get('path') else ''
    if record_root is not None:
        for ep in episodes:
            if 'td_advantage' in ep:
                ep['advantage_indicator']=ep['td_advantage']>cfg['train']['threshold']
                ep.update(value_checkpoint_sha256=np.array(value_digest),
                          advantage_threshold=np.array(cfg['train']['threshold']),n_step=np.array(cfg['train']['n_step']),
                          return_scale=np.array(cfg['env']['horizon']+149))
            ep['value_alignment']=np.array('observation_before_action')
            ep.update(policy_checkpoint_sha256=np.array(policy_digest),policy_checkpoint_step=np.array(policy_info.get('step') or -1))
            store(record_root,'evaluation',record_source,ep)
    return dict(metrics=episode_metrics(episodes), successes={s['scene_id']: bool(e['success'][-1]) for s,e in zip(scenes,episodes)})


def record_annotations(root, episodes, origins, cfg, round_index):
    """Compact per-round labels for the cumulative replay, linked to raw NPZs."""
    root=Path(root)
    critic=root/'value.pt'
    digest=sha256(critic.read_bytes()).hexdigest()
    diagnostics=json.loads((root/'diagnostics.json').read_text()) if (root/'diagnostics.json').is_file() else {}
    threshold=diagnostics.get('threshold',cfg['train']['threshold'])
    rows=[]
    for ep in episodes:
        identifier=str(ep['episode_id'])
        origin=Path(origins[identifier])
        if not origin.is_file(): raise FileNotFoundError(f'Raw trajectory missing: {origin}')
        kind='corrections' if np.asarray(ep.get('intervention_mask',False)).any() else 'rollout'
        directory=root/'data/adaptation/annotations'/kind
        directory.mkdir(parents=True,exist_ok=True)
        fields={key:np.asarray(ep[key]) for key in ('value_prediction','mc_return','td_advantage','advantage_indicator')}
        length=len(ep['action'])
        if any(array.shape!=(length,) for array in fields.values()):
            raise ValueError('Trajectory annotations must align one-to-one with actions')
        fields.update(episode_id=ep['episode_id'],scene_id=ep['scene_id'],frame_index=np.arange(length),
                      intervention_mask=np.asarray(ep.get('intervention_mask',np.zeros(length,dtype=bool))),
                      trajectory_path=np.array(Path(os.path.relpath(origin,directory)).as_posix()),
                      value_checkpoint=np.array(Path(os.path.relpath(critic,directory)).as_posix()),
                      value_checkpoint_sha256=np.array(digest),annotation_round=np.array(round_index),
                      advantage_threshold=np.array(threshold),n_step=np.array(cfg['train']['n_step']),
                      return_scale=np.array(cfg['env']['horizon']+149),
                      value_alignment=np.array('observation_before_action'))
        offset=int(ep.get('control_step_offset',0))
        fields.update(control_step_offset=np.array(offset),task_step_index=np.arange(length)+offset,
                      within_task_horizon=np.arange(length)+offset<cfg['env']['horizon'],
                      value_supervision_steps=np.array(min(length,max(0,cfg['env']['horizon']-offset))),
                      mc_return_definition=np.array('full_recorded_segment_return'))
        # Enrich only this round's newly collected NPZs. Older raw trajectories
        # keep their original round's labels; sidecars retain all later relabels.
        if origin.resolve().is_relative_to(root.resolve()):
            from .cli import store
            enriched=dict(ep)
            for key in ('value_checkpoint_sha256','annotation_round','advantage_threshold','n_step',
                        'return_scale','value_alignment','value_supervision_steps','within_task_horizon'):
                enriched[key]=fields[key]
            store(root,'adaptation',kind,enriched)
        destination=directory/f'{identifier}.npz'
        np.savez_compressed(destination,**fields)
        rows.append(dict(episode_id=identifier,scene_id=str(ep['scene_id']),frames=length,
                         trajectory=str(origin.relative_to(root.parent)),annotations=str(destination.relative_to(root))))
    write_json(root/'annotations.json',dict(round=round_index,value_checkpoint_sha256=digest,
               alignment='index t corresponds to observation[t] and action[t]; raw qpos[t] is after that action',
               trajectories=rows))


def collect(model, scenes, cfg, count, seed, root):
    from .cli import policy, store
    rng = np.random.default_rng(seed)
    env = ObstacleReach2D(cfg['env'])
    episodes, failures = [], []
    info=getattr(model,'checkpoint_info',{})
    digest=sha256(Path(info['path']).read_bytes()).hexdigest() if info.get('path') else ''
    try:
        for i in tqdm(range(count),desc='Autonomous practice',unit='episode',dynamic_ncols=True):
            scene = scenes[int(rng.integers(len(scenes)))]
            ep = rollout(env, scene, policy(model), seed+i, 'practice', cfg['data']['exploration'])
            ep.update(scene_json=np.array(json.dumps(scene)), intervention_mask=np.zeros(len(ep['action']),dtype=bool))
            ep.update(policy_checkpoint_sha256=np.array(digest),policy_checkpoint_step=np.array(info.get('step') or -1))
            store(root,'adaptation','rollout',ep)
            episodes.append(ep)
            if not ep['success'][-1]: failures.append((scene,ep))
    finally: env.close()
    return episodes, failures


def run_experiment(source, output, modes, splits, options):
    from .cli import fit, load_model, policy, store
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists(): raise FileExistsError(f'Experiment already exists: {output}; choose another --experiment name')
    if not output.is_relative_to(source): raise ValueError('Improvement results must stay inside the source run')
    frozen = json.loads((source/'config.json').read_text())
    manifest = json.loads((source/'scenes.json').read_text())
    initial_bytes=(source/'bc.pt').read_bytes()
    ck = torch.load(BytesIO(initial_bytes),map_location='cpu',weights_only=False)
    check_environment(frozen,ck['config'])
    cfg = copy.deepcopy(frozen)
    cfg['device'] = options.device
    cfg['seed'] = options.seed
    cfg['data']['exploration'] = options.noise
    cfg['train'] = copy.deepcopy(ck['config']['train'])
    if options.train_config:
        overrides=yaml.safe_load(Path(options.train_config).read_text())['train']
        for key in ('width','depth'):
            if key in overrides and overrides[key]!=cfg['train'][key]:
                raise ValueError('Improvement fine-tuning must keep the initial BC architecture')
        cfg['train'].update(overrides)
    cfg['train'].update(max_steps=options.steps,lr=options.lr,validation_steps=options.steps,
                        checkpoint_steps=options.steps,intervention_weight=options.correction_weight)
    target,pool_sources=select_scene_pool(manifest,splits,options.scene_ids)
    adapt, holdout = partition_scenes(target, options.holdout_fraction, options.seed)
    output.mkdir(parents=True)
    initial_checkpoint=output/'initial.bc.pt'
    initial_checkpoint.write_bytes(initial_bytes)
    baseline = load_model(source,'bc',cfg,checkpoint=initial_checkpoint)
    baseline_hash = sha256(initial_bytes).hexdigest()
    protocol = dict(config=cfg, modes=modes, options=vars(options), initial_bc_sha256=baseline_hash,
                    evaluation_protocol='pooled_practice' if not holdout else 'practice_with_holdout',
                    scene_sampling='uniform_with_replacement',
                    policy_update='always_use_latest_round',
                    scene_pool_sources=pool_sources,scene_pool_size=len(target),
                    practice_scene_ids=[s['scene_id'] for s in adapt],
                    adaptation_scene_ids=[s['scene_id'] for s in adapt],
                    holdout_scene_ids=[s['scene_id'] for s in holdout],
                    original_train_scenes=len(manifest['train']), original_validation_scenes=len(manifest['validation']),
                    original_training_sources=['data/train/expert','data/train/rollout'],
                    original_rollout_episodes=len(list((source/'data/train/rollout').glob('*.npz'))),
                    note='Both arms train on original train data plus uniform autonomous practice. Corrections adds expert recovery segments without a control-step cap. Value keeps the original task deadline; full recovery segments supervise the policy. Optional target holdout is never used for fitting or checkpoint selection.')
    write_json(output/'protocol.json',protocol)
    initial = evaluate_scenes(baseline,adapt,cfg,options.seed+80000000,'Initial practice pool regression')
    validation = evaluate_scenes(baseline,manifest['validation'],cfg,options.seed+81000000,'Initial source validation')
    initial_holdout = evaluate_scenes(baseline,holdout,cfg,options.seed+82000000,'Initial target holdout')
    write_json(output/'baseline.json',dict(adaptation=initial,validation=validation,holdout=initial_holdout))
    summaries = {}
    for mode in modes:
        active, active_value, active_path = baseline, None, str(initial_checkpoint)
        active_kind='bc'
        previous, previous_validation = initial, validation
        experience, corrections, value_corrections, origins = [], [], [], {}
        history = []
        arm = output/mode; arm.mkdir()
        for round_index in range(1,options.rounds+1):
            # Identical seeds/initial weights/data for both arms. Later behavior
            # can differ as their learned policies evolve.
            seed = options.seed+round_index*1000000
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            root = arm/f'round-{round_index:03d}'; root.mkdir()
            write_json(root/'config.json',frozen)
            write_json(root/'scenes.json',manifest)
            write_json(root/'training_input.json',dict(expert_root=str(source),active_policy=active_path,
                       includes_original_training_data=True,source_validation_is_training_holdout=True))
            episodes, failures = collect(active,adapt,cfg,options.episodes,seed,root)
            experience.extend(episodes)
            for ep in episodes:
                origins[str(ep['episode_id'])]=root/'data/adaptation/rollout'/f"{ep['episode_id']}.npz"
            budget = dict(autonomous_episodes=len(episodes),autonomous_steps=sum(len(e['action']) for e in episodes),
                          expert_steps=0,replay_steps=0,expert_attempts=0,correction_episodes=0,policy_suffix_steps=0,
                          expert_labeled_steps=0,assisted_successes=0,expert_recovery_successes=0)
            interventions = []
            if mode=='corrections':
                failures.sort(key=lambda pair:float(pair[1]['distance'][-1]))
                for index,(scene,failed) in enumerate(failures[:options.expert_episodes]):
                    ep, result = takeover_episode(scene,failed,cfg,seed+500000+index,options.lookback)
                    budget['expert_attempts']+=1; budget['expert_steps']+=result['expert_steps']
                    budget['replay_steps']+=result['replay_steps']
                    budget['policy_suffix_steps']+=result.get('policy_suffix_steps',0)
                    interventions.append(dict(parent_episode=str(failed['episode_id']),scene_id=scene['scene_id'],**result))
                    if ep is not None:
                        ep.update(parent_policy_sha256=failed['policy_checkpoint_sha256'])
                        budget['expert_labeled_steps']+=int(ep['intervention_mask'].sum())
                        budget['assisted_successes']+=int(ep['task_success'])
                        budget['expert_recovery_successes']+=1
                        corrections.append(ep); budget['correction_episodes']+=1
                        value_corrections.append(correction_value_episode(ep,cfg))
                        store(root,'adaptation','corrections',ep)
                        origins[str(ep['episode_id'])]=root/'data/adaptation/corrections'/f"{ep['episode_id']}.npz"
            extras = experience+corrections
            fit(root,'value',cfg,expert_root=source,extra_train=experience+value_corrections,initial_value=active_value,persist_annotations=False)
            fit(root,'recap',cfg,expert_root=source,extra_train=extras,initial_policy=active,persist_annotations=False)
            record_annotations(root,extras,origins,cfg,round_index)
            metric_cfg=copy.deepcopy(cfg)
            metric_cfg['train']['threshold']=json.loads((root/'diagnostics.json').read_text())['threshold'] if (root/'diagnostics.json').is_file() else cfg['train']['threshold']
            candidate=load_model(root,'recap',cfg,checkpoint=root/'recap.last.pt')
            active=candidate;active_value=load_model(root,'value',cfg);active_path=str(root/'recap.last.pt')
            active_kind='recap'
            assessment=evaluate_scenes(candidate,adapt,metric_cfg,options.seed+80000000,'Practice metrics (report only)',
                        record_root=root,record_source='practice',value=active_value)
            val=evaluate_scenes(candidate,manifest['validation'],metric_cfg,options.seed+81000000,'Source validation metrics (report only)',
                        record_root=root,record_source='validation',value=active_value)
            changes=regression_changes(previous['successes'],assessment['successes'],previous_validation['successes'],val['successes'])
            result=dict(round=round_index,mode=mode,budget=budget,interventions=interventions,policy_updated=True,changes=changes,
                        candidate_adaptation=assessment,candidate_validation=val,
                        cumulative_autonomous_episodes=len(experience),cumulative_correction_episodes=len(corrections),
                        original_training_data_included=True)
            previous,previous_validation=assessment,val
            result['active_policy']=active_path
            result['active_policy_kind']=active_kind
            result['active_adaptation']=previous
            history.append(result);write_json(root/'result.json',result);write_json(arm/'history.json',history)
            print(f"{mode} round {round_index}: policy updated; recovered={len(changes['recovered'])}; regressions={len(changes['regressions'])}",flush=True)
        final_holdout=evaluate_scenes(active,holdout,cfg,options.seed+82000000,'Final target holdout (report only)')
        summaries[mode]=dict(active_policy=active_path,active_policy_kind=active_kind,adaptation=previous,holdout=final_holdout,rounds=history)
        write_json(arm/'summary.json',summaries[mode])
    write_json(output/'comparison.json',dict(protocol=protocol,baseline=initial,baseline_holdout=initial_holdout,arms=summaries))
    return summaries


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True)
    p.add_argument('--mode',choices=['autonomous','corrections','both'],default='both')
    p.add_argument('--experiment',default='paired-v1')
    p.add_argument('--splits',nargs='+',choices=['all','train',*TEST_SPLITS,'goal_ood','composition_ood'],default=['all'],
                   help='Scene sources to pool; default all merges every source except validation.')
    p.add_argument('--scene-ids',nargs='+')
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--episodes',type=int,default=200)
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--lr',type=float,default=.0001)
    p.add_argument('--noise',type=float,default=.005)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='auto')
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--holdout-fraction',type=float,default=0.,help='Optional separate holdout; default 0 practices the whole pool.')
    p.add_argument('--expert-episodes',type=int,default=10)
    p.add_argument('--lookback',type=int,default=25)
    p.add_argument('--correction-weight',type=float,default=5.)
    p.add_argument('--train-config')
    args=p.parse_args(argv)
    if min(args.rounds,args.episodes,args.steps,args.threads,args.lookback)<=0:
        p.error('rounds, episodes, steps, threads and lookback must be positive')
    if args.expert_episodes<0: p.error('expert-episodes must be nonnegative')
    if not 0<=args.holdout_fraction<1: p.error('holdout-fraction must be in [0,1)')
    if not all(np.isfinite(x) for x in (args.lr,args.noise,args.correction_weight)) or args.lr<=0 or args.noise<0 or args.correction_weight<1:
        p.error('invalid learning rate, noise or replay weights')
    if Path(args.experiment).name!=args.experiment or args.experiment in ('','.','..'):
        p.error('experiment must be a single directory name')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
    modes=['autonomous','corrections'] if args.mode=='both' else [args.mode]
    run_experiment(args.run,Path(args.run)/'improvement'/args.experiment,modes,args.splits,args)


if __name__=='__main__': main()
