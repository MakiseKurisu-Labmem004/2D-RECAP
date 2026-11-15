import argparse
import copy
import json
from pathlib import Path
import random
import numpy as np
import torch
import yaml
from tqdm import tqdm
from .env import ObstacleReach2D
from .planning import rollout
from .models import Model,returns,advantage,distribution_targets
from .datasets import DATASET_SPLITS
from .generation import generate_datasets,write_json
from .evaluation import episode_metrics,difficulty_metrics
from .observations import OBSERVATION_DIM,convert_episode
from .training import make_optimizer,train_steps,positive_integer,save_checkpoint
from .goal_supervision import goal_anchors


def save_json(path,obj):
    write_json(path,obj)


def episodes(root,split,source='expert'):
    paths=sorted((root/'data'/split/source).glob('*.npz'))
    with tqdm(paths,desc=f'Load {split}/{source}',unit='episode',leave=False,dynamic_ncols=True) as progress:
        result=[]
        for path in progress:
            with np.load(path,allow_pickle=False) as archive:
                result.append(convert_episode(dict(archive)))
        return result


def store(root,split,source,ep):
    p=root/'data'/split/source; p.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(p/f"{ep['episode_id']}.npz",**convert_episode(ep))


def device(cfg):
    return ('cuda' if torch.cuda.is_available() else 'cpu') if cfg['device']=='auto' else cfg['device']


def resolve_checkpoint(root,checkpoint):
    path=Path(checkpoint)
    candidates=[path] if path.is_absolute() else [Path(root)/path]
    if not path.is_absolute():
        if path.parent==Path('.'): candidates.append(Path(root)/'checkpoints'/path)
        candidates.append(path)
    for candidate in candidates:
        if candidate.is_file(): return candidate.resolve()
    raise FileNotFoundError(f'Checkpoint not found: {checkpoint}; tried '+', '.join(str(p) for p in candidates))


def checkpoint_label(name,path):
    label=path.stem
    if label==name: label=f'{name}.checkpoint'
    elif not label.startswith(f'{name}.'): label=f'{name}.{label}'
    return ''.join(c if c.isalnum() or c in '._-' else '_' for c in label)


def load_model(root,name,cfg,checkpoint=None):
    path=Path(checkpoint) if checkpoint is not None else root/f'{name}.pt'
    ck=torch.load(path,map_location=device(cfg),weights_only=False)
    check_checkpoint_observations(ck,name)
    model_cfg=ck.get('config',cfg)
    if checkpoint is not None and model_cfg['env']!=cfg['env']:
        raise ValueError('Checkpoint environment differs from this run; evaluate with its matching run/config.json')
    model=Model(name,model_cfg['train'],model_cfg['env']['action_limit']).to(device(cfg))
    model.load_state_dict(ck['model']); model.eval()
    model.checkpoint_info=dict(path=str(path.resolve()),step=ck.get('global_step'),epoch=ck.get('epoch'))
    return model


def check_checkpoint_observations(checkpoint,kind):
    if checkpoint.get('model_kind',kind)!=kind:
        raise ValueError(f'Checkpoint contains {checkpoint["model_kind"]}, but --model is {kind}')
    expected=OBSERVATION_DIM+(3 if kind=='recap' else 0)
    if checkpoint['model']['net.0.weight'].shape[1]!=expected:
        raise ValueError(f'{kind} checkpoint uses the old observation layout. '
                         'Retrain with 9D observations; use recap migrate to copy existing data to a new run.')


def policy(model):
    def act(obs,env):
        with torch.no_grad(): return model.action(torch.as_tensor(obs,device=next(model.parameters()).device)[None])[0].cpu().numpy()
    return act


def annotate(eps,model,cfg):
    scale=cfg['env']['horizon']+149
    with tqdm(eps,desc='Value/advantage',unit='episode',leave=False,dynamic_ncols=True) as progress:
        for ep in progress:
            with torch.no_grad():
                ep['value_prediction']=model.value(torch.tensor(ep['observation'],device=device(cfg))).cpu().numpy()
                nv=model.value(torch.tensor(ep['next_observation'],device=device(cfg))).cpu().numpy()
            ep['mc_return']=returns(ep['reward'],scale)
            ep['td_advantage']=advantage(ep['reward'],ep['value_prediction'],nv,cfg['train']['n_step'],scale)
    return eps


def fit(root,kind,cfg,resume=False):
    cfg=copy.deepcopy(cfg)
    tc=cfg['train']; dev=device(cfg)
    last=root/f'{kind}.last.pt'
    checkpoint=torch.load(last,map_location=dev,weights_only=False) if resume and last.exists() else None
    if checkpoint: check_checkpoint_observations(checkpoint,kind)
    source=load_model(root,'bc',cfg) if kind=='sft' else None
    saved_train=checkpoint.get('config',{}).get('train',{}) if checkpoint else {}
    tc.pop('policy_use_velocity',None)
    step_mode=tc.get('max_steps') is not None
    if step_mode: positive_integer(tc['max_steps'],'max_steps')
    if checkpoint and step_mode!=('train_state' in checkpoint):
        raise ValueError('Resume with the checkpoint training budget mode (steps or epochs)')
    interval_default=saved_train.get('validation_rollout_interval',0) if checkpoint else 10
    if kind=='bc' and checkpoint and tc.get('validation_rollout_interval',interval_default)!=interval_default:
        raise ValueError('Resume with the checkpoint validation_rollout_interval setting')
    validation_interval=tc.get('validation_rollout_interval',interval_default) if kind=='bc' else 0
    if not isinstance(validation_interval,int) or isinstance(validation_interval,bool) or validation_interval<0:
        raise ValueError('validation_rollout_interval must be a nonnegative integer')
    tc['validation_rollout_interval']=validation_interval
    train=episodes(root,'train'); val=episodes(root,'validation')
    if kind!='bc':
        train+=episodes(root,'train','rollout'); val+=episodes(root,'validation','rollout')
    if kind=='sft':
        train=[ep for ep in train if ep['success'][-1]]
        val=[ep for ep in val if ep['success'][-1]]
    if not train or not val: raise RuntimeError('Generate training and validation episodes first')
    threshold=tc['threshold']
    if kind=='recap':
        value=load_model(root,'value',cfg)
        annotate(train,value,cfg); annotate(val,value,cfg)
        adv=np.concatenate([ep['td_advantage'] for ep in train])
        if tc['selection']=='quantile': threshold=float(np.quantile(adv,tc['quantile']))
        for split,eps in [('train',train),('validation',val)]:
            with tqdm(eps,desc=f'Label {split}',unit='episode',leave=False,dynamic_ncols=True) as progress:
                for ep in progress:
                    ep['advantage_indicator']=ep['td_advantage']>threshold
                    store(root,split,'annotated',ep)
        save_json(root/'diagnostics.json',dict(transitions=len(adv),positive_fraction=float((adv>0).mean()),
            selected_fraction=float((adv>threshold).mean()),threshold=threshold,
            successful_rollouts=sum(bool(e['success'][-1]) for e in episodes(root,'train','rollout')),
            note='Increase rollout budget if positive or successful coverage is insufficient; no automatic performance claim.'))
    def tensors(eps):
        x=np.concatenate([e['observation'] for e in eps])
        y=np.concatenate([returns(e['reward'],cfg['env']['horizon']+149) if kind=='value' else e['action'] for e in eps])
        label=np.concatenate([e['advantage_indicator'] for e in eps]) if kind=='recap' else np.zeros(len(x))
        return tuple(torch.tensor(a,dtype=torch.float32,device=dev) for a in (x,y,label))
    x,y,labels=tensors(train); vx,vy,vl=tensors(val)
    anchor_count=tc.get('goal_anchor_samples',0) if kind=='bc' else 0
    if anchor_count:
        manifest=json.loads((root/'scenes.json').read_text())
        ax,ay=goal_anchors(manifest['train'],cfg)
        x=torch.cat([x,torch.as_tensor(ax,device=dev)])
        y=torch.cat([y,torch.as_tensor(ay,device=dev)])
        labels=torch.cat([labels,torch.zeros(len(ax),device=dev)])
        save_json(root/'bc.goal_anchors.json',dict(samples=len(ax),scenes=len(manifest['train']),radius=tc.get('goal_anchor_radius',.12),
                  note='Training-only local expert labels; no evaluation correction or altered success threshold.'))
        tqdm.write(f'BC goal anchors: {len(ax)} training-only samples')
    model=Model(kind,tc,cfg['env']['action_limit']).to(dev)
    if source is not None: model.load_state_dict(source.state_dict())
    optimizer=make_optimizer(model,tc,saved_train if checkpoint else None)
    start=0; best=float('inf'); history=[]
    best_success=-1.
    if checkpoint:
        ck=checkpoint; model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
        start=ck['epoch']+1; best=ck['best']; history=ck['history']
        best_success=ck.get('best_success',-1.)
        torch.set_rng_state(ck['rng'].cpu())
        if dev.startswith('cuda') and ck.get('cuda_rng') is not None: torch.cuda.set_rng_state_all(ck['cuda_rng'])
    def loss(a,b,l,training):
        if kind=='value':
            return ((model.value(a)-b)**2).mean() if tc['scalar_value'] else -(distribution_targets(b,tc['bins'])*model(a).log_softmax(-1)).sum(-1).mean()
        if kind!='recap': return ((model(a)-b)/cfg['env']['action_limit']).square().mean()
        null=torch.full((len(a),),2,dtype=torch.long,device=dev)
        condition=l.long()
        if training: condition=torch.where(torch.rand(len(a),device=dev)<tc['dropout'],null,condition)
        return model.nll(a,b,null).mean()+tc['alpha']*(l*model.nll(a,b,condition)).mean()+tc['low_weight']*((1-l)*model.nll(a,b,condition)).mean()
    validation_scenes=[]
    if validation_interval or (step_mode and kind=='bc'):
        manifest=json.loads((root/'scenes.json').read_text())
        validation_scenes=manifest.get('validation',[])
        if not validation_scenes: raise ValueError('BC closed-loop validation requires validation scenes')
    if step_mode:
        train_steps(root,kind,cfg,model,optimizer,(x,y,labels),(vx,vy,vl),loss,validation_scenes,policy,checkpoint)
        return
    tqdm.write(f'{kind} observation_dim={OBSERVATION_DIM}, validation_rollout_interval={validation_interval}')
    with tqdm(range(start,tc['epochs']),total=tc['epochs'],initial=start,
              desc=f'Train {kind}',unit='epoch',dynamic_ncols=True) as epochs:
        for epoch in epochs:
            model.train(); total=0.
            batches=torch.randperm(len(x),device=dev).split(tc['batch_size'])
            with tqdm(batches,desc=f'{kind} epoch {epoch+1}',unit='batch',leave=False,dynamic_ncols=True) as progress:
                for ids in progress:
                    optimizer.zero_grad(); err=loss(x[ids],y[ids],labels[ids],True); err.backward(); optimizer.step(); total+=err.item()*len(ids)
            model.eval()
            with torch.no_grad(): validation=loss(vx,vy,vl,False).item()
            history.append(dict(epoch=epoch,train=total/len(x),validation=validation))
            improved=validation<best
            if validation_interval:
                improved=False
                if epoch==0 or (epoch+1)%validation_interval==0 or epoch+1==tc['epochs']:
                    env=ObstacleReach2D(cfg['env'])
                    try:
                        with tqdm(validation_scenes,desc='BC validation rollout',unit='episode',leave=False,dynamic_ncols=True) as progress:
                            results=[rollout(env,scene,policy(model),cfg['seed']+30000000+i,'bc-validation')
                                     for i,scene in enumerate(progress)]
                    finally: env.close()
                    metrics=episode_metrics(results)
                    success=metrics['success_rate']
                    history[-1].update(validation_success_rate=success,validation_collision_rate=metrics['collision_rate'],
                                       validation_timeout_rate=metrics['timeout_rate'],
                                       validation_by_difficulty=difficulty_metrics(results,validation_scenes))
                    if 'strict_success_rate' in metrics:
                        history[-1].update(validation_strict_success_rate=metrics['strict_success_rate'],
                                           validation_tolerance_only_success_count=metrics['tolerance_only_success_count'])
                    improved=success>best_success or (success==best_success and validation<best)
                    if improved: best_success=success; best=validation
            else: best=min(best,validation)
            ck=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,best=best,history=history,
                    best_success=best_success,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    config=cfg,model_kind=kind)
            save_checkpoint(last,ck)
            if improved: save_checkpoint(root/f'{kind}.pt',ck)
            epochs.set_postfix(train=f'{total/len(x):.5f}',validation=f'{validation:.5f}')
            tqdm.write(f'{kind} {history[-1]}')
    save_json(root/f'{kind}.history.json',history)


def evaluate_policies(root,cfg,models=None,splits=None,checkpoint=None):
    root=Path(root)
    models=['bc','sft','recap'] if models is None else models
    if checkpoint is not None and len(models)!=1:
        raise ValueError('A checkpoint requires exactly one policy model')
    selected_checkpoint=resolve_checkpoint(root,checkpoint) if checkpoint is not None else None
    save=selected_checkpoint is None
    scenes=json.loads((root/'scenes.json').read_text())
    selected=list(dict.fromkeys(splits)) if splits is not None else [name for name in scenes if name not in ('train','validation') and scenes[name]]
    if not selected or any(name not in scenes or not scenes[name] for name in selected):
        raise ValueError('Select nonempty datasets present in scenes.json')
    for name in models:
        if name not in ('bc','sft','recap'):
            raise ValueError('Evaluate supports bc, sft, and recap policies')
        if selected_checkpoint is None and not (root/f'{name}.pt').is_file():
            raise FileNotFoundError(root/f'{name}.pt')
    chosen=load_model(root,models[0],cfg,checkpoint=selected_checkpoint) if selected_checkpoint is not None else None
    if chosen is not None:
        info=getattr(chosen,'checkpoint_info',{})
        tqdm.write(f'Checkpoint: {selected_checkpoint}'+(f" (step {info['step']})" if info.get('step') is not None else ''))
    value=load_model(root,'value',cfg) if (root/'value.pt').is_file() else None
    report={}; env=ObstacleReach2D(cfg['env'])
    try:
        for name in models:
            model=chosen if chosen is not None else load_model(root,name,cfg)
            label=checkpoint_label(name,selected_checkpoint) if selected_checkpoint is not None else name
            report[name]={}; all_episodes=[]
            for split in selected:
                with tqdm(scenes[split],desc=f'Evaluate {label}/{split}',unit='episode',dynamic_ncols=True) as progress:
                    eps=[rollout(env,scene,policy(model),cfg['seed']+20000000+i,label) for i,scene in enumerate(progress)]
                if value is not None: annotate(eps,value,cfg)
                if save:
                    for ep in eps: store(root,split,label,ep)
                metrics=episode_metrics(eps)
                metrics['by_difficulty']=difficulty_metrics(eps,scenes[split])
                if selected_checkpoint is not None:
                    metrics.update(checkpoint=str(selected_checkpoint),checkpoint_step=getattr(model,'checkpoint_info',{}).get('step'))
                report[name][split]=metrics
                all_episodes.extend(eps)
                tqdm.write(f"{label}/{split} | success {metrics['success_count']}/{metrics['episodes']} ({metrics['success_rate']:.1%})")
                if 'strict_success_count' in metrics and float(eps[0]['strict_success_distance']) < float(eps[0]['success_distance']):
                    tqdm.write(f"  strict {float(eps[0]['strict_success_distance'])*100:g} cm | success "
                               f"{metrics['strict_success_count']}/{metrics['episodes']} ({metrics['strict_success_rate']:.1%}); "
                               f"tolerance-only {metrics['tolerance_only_success_count']}")
                for difficulty,group in metrics['by_difficulty'].items():
                    tqdm.write(f"  {difficulty} | {group['success_count']}/{group['episodes']} ({group['success_rate']:.1%})")
            if name=='recap' and 'bc' in report:
                for split in selected:
                    report[name][split]['gain_over_bc']=report[name][split]['success_rate']-report['bc'][split]['success_rate']
            if save: save_json(root/f'evaluation.{label}.json',{name:report[name]})
            overall=episode_metrics(all_episodes)
            tqdm.write(f"{label} overall | success {overall['success_count']}/{overall['episodes']} ({overall['success_rate']:.1%})")
        if save and len(models)>1: save_json(root/'evaluation.json',report)
    finally:
        env.close()
    return report


def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['generate','train','rollout','evaluate','plot','all','migrate'])
    p.add_argument('--config',help='Configuration file; evaluate defaults to the saved run config, other commands to configs/default.yaml')
    p.add_argument('--run',default='runs/default')
    p.add_argument('--model',choices=['bc','sft','value','recap'],
                   help='Train this model or evaluate this policy. Defaults to bc for train and all policies for evaluate.')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--checkpoint',help='For evaluate --model: checkpoint path, relative to the run or absolute; bare names also search run/checkpoints')
    p.add_argument('--output-run',help='Destination for migrate: a new run with 9D trajectory data and no old checkpoints.')
    p.add_argument('--lr',type=float,help='Learning rate in the new run created by migrate.')
    p.add_argument('--train-config',help='For migrate: merge the train section of this YAML into the new run configuration.')
    p.add_argument('--splits',nargs='+',choices=(*DATASET_SPLITS,'goal_ood','composition_ood'),
                   help='Generate or evaluate only these datasets. Combine generate with --resume to skip completed datasets.')
    args=p.parse_args()
    if args.checkpoint and (args.command!='evaluate' or not args.model):
        p.error('--checkpoint requires evaluate and an explicit --model')
    if args.command=='migrate':
        if not args.output_run: p.error('migrate requires --output-run')
        if args.model or args.splits or args.resume: p.error('migrate does not support --model, --splits, or --resume')
        from .migration import migrate_run
        overrides=yaml.safe_load(Path(args.train_config).read_text())['train'] if args.train_config else None
        migrate_run(args.run,args.output_run,lr=args.lr,train_overrides=overrides)
        return
    if args.output_run or args.lr is not None or args.train_config:
        p.error('--output-run, --lr, and --train-config are supported only by migrate')
    root=Path(args.run)
    config_path=Path(args.config) if args.config else root/'config.json' if args.command=='evaluate' and (root/'config.json').is_file() else Path('configs/default.yaml')
    cfg=yaml.safe_load(config_path.read_text())
    if args.command!='evaluate': root.mkdir(parents=True,exist_ok=True)
    if args.splits and args.command not in ('generate','evaluate','all'):
        p.error('--splits is supported by generate, evaluate, and all')
    if args.command=='evaluate' and args.model=='value':
        p.error('evaluate --model requires a policy model such as bc, sft, or recap')
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    torch.use_deterministic_algorithms(True)
    if (root/'config.json').exists() and json.loads((root/'config.json').read_text())!=cfg:
        raise ValueError('Use a new run directory for changed configurations')
    if args.command not in ('generate','all','evaluate'):
        save_json(root/'config.json',cfg)
    def generate():
        generate_datasets(root,cfg,selected=args.splits,resume=args.resume)
    def collect():
        scenes=json.loads((root/'scenes.json').read_text()); env=ObstacleReach2D(cfg['env']); model=load_model(root,'bc',cfg)
        try:
            for j,split in enumerate(['train','validation']):
                count=cfg['data']['rollouts'] if split=='train' else len(scenes[split])
                with tqdm(range(count),desc=f'Rollout {split}',unit='episode',dynamic_ncols=True) as progress:
                    for i in progress:
                        ep=rollout(env,scenes[split][i%len(scenes[split])],policy(model),cfg['seed']+10000000+j*100000+i,'bc',cfg['data']['exploration'])
                        store(root,split,'rollout',ep)
        finally: env.close()
    def evaluate():
        models=[args.model] if args.command=='evaluate' and args.model else None
        selected=args.splits if args.command=='evaluate' else None
        evaluate_policies(root,cfg,models=models,splits=selected,checkpoint=args.checkpoint)
    def plot():
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        eps=episodes(root,'id','recap')
        paths=list(root.glob('*.history.json'))
        with tqdm(total=len(paths)+bool(eps),desc='Plot',unit='figure',dynamic_ncols=True) as progress:
            for path in paths:
                h=json.loads(path.read_text()); fig,ax=plt.subplots()
                for key in ['train','validation']: ax.plot([r[key] for r in h],label=key)
                ax.legend(); fig.savefig(root/f'{path.stem}.png'); plt.close(fig)
                progress.update(1)
            if eps:
                ep=eps[0]; fig,axes=plt.subplots(1,3,figsize=(15,4))
                for key in ['value_prediction','mc_return','td_advantage']: axes[0].plot(ep[key],label=key)
                axes[0].legend(); axes[1].scatter(ep['mc_return'],ep['value_prediction'],s=5)
                from .env import points
                scenes=json.loads((root/'scenes.json').read_text()); scene=next(s for s in scenes['id'] if s['scene_id']==str(ep['scene_id']))
                axes[2].add_patch(plt.Circle(scene['obstacle'],scene['radius'],color='red'))
                for q in ep['qpos'][::5]:
                    ps=points(q); axes[2].plot(ps[:,0],ps[:,1],alpha=.25)
                axes[2].scatter(*scene['goal'],marker='*'); axes[2].set(xlim=(-1,1),ylim=(-1,1),aspect='equal')
                fig.savefig(root/'trajectory.png'); plt.close(fig)
                progress.update(1)
    if args.command=='all':
        stages=[('generate',generate),('train bc',lambda: fit(root,'bc',cfg)),('rollout',collect),
                ('train sft',lambda: fit(root,'sft',cfg)),('train value',lambda: fit(root,'value',cfg)),
                ('train recap',lambda: fit(root,'recap',cfg)),('evaluate',evaluate),('plot',plot)]
        with tqdm(stages,desc='All stages',unit='stage',dynamic_ncols=True) as progress:
            for name,action in progress:
                progress.set_postfix_str(name)
                action()
    elif args.command=='generate': generate()
    elif args.command=='train': fit(root,args.model or 'bc',cfg,args.resume)
    elif args.command=='rollout': collect()
    elif args.command=='evaluate': evaluate()
    else: plot()

if __name__=='__main__': main()
