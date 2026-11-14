import argparse
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


def save_json(path,obj):
    write_json(path,obj)


def episodes(root,split,source='expert'):
    paths=sorted((root/'data'/split/source).glob('*.npz'))
    with tqdm(paths,desc=f'Load {split}/{source}',unit='episode',leave=False,dynamic_ncols=True) as progress:
        return [dict(np.load(p,allow_pickle=False)) for p in progress]


def store(root,split,source,ep):
    p=root/'data'/split/source; p.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(p/f"{ep['episode_id']}.npz",**ep)


def device(cfg):
    return ('cuda' if torch.cuda.is_available() else 'cpu') if cfg['device']=='auto' else cfg['device']


def load_model(root,name,cfg):
    ck=torch.load(root/f'{name}.pt',map_location=device(cfg),weights_only=False)
    model=Model(name,cfg['train'],cfg['env']['action_limit']).to(device(cfg))
    model.load_state_dict(ck['model']); model.eval(); return model


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
    tc=cfg['train']; dev=device(cfg)
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
    model=Model(kind,tc,cfg['env']['action_limit']).to(dev)
    if kind=='sft': model.load_state_dict(load_model(root,'bc',cfg).state_dict())
    optimizer=torch.optim.Adam(model.parameters(),lr=tc['lr']); start=0; best=float('inf'); history=[]
    last=root/f'{kind}.last.pt'
    if resume and last.exists():
        ck=torch.load(last,map_location=dev,weights_only=False); model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
        start=ck['epoch']+1; best=ck['best']; history=ck['history']
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
            improved=validation<best; best=min(best,validation)
            ck=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,best=best,history=history,
                    rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,config=cfg)
            torch.save(ck,last)
            if improved: torch.save(ck,root/f'{kind}.pt')
            epochs.set_postfix(train=f'{total/len(x):.5f}',validation=f'{validation:.5f}')
            tqdm.write(f'{kind} {history[-1]}')
    save_json(root/f'{kind}.history.json',history)


def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['generate','train','rollout','evaluate','plot','all'])
    p.add_argument('--config',default='configs/default.yaml'); p.add_argument('--run',default='runs/default')
    p.add_argument('--model',choices=['bc','sft','value','recap'],default='bc'); p.add_argument('--resume',action='store_true')
    p.add_argument('--splits',nargs='+',choices=(*DATASET_SPLITS,'goal_ood','composition_ood'),
                   help='Generate only these datasets. Combine with --resume to skip completed datasets.')
    args=p.parse_args(); cfg=yaml.safe_load(Path(args.config).read_text()); root=Path(args.run); root.mkdir(parents=True,exist_ok=True)
    if args.splits and args.command not in ('generate','all'):
        p.error('--splits is supported by generate and all')
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    torch.use_deterministic_algorithms(True)
    if (root/'config.json').exists() and json.loads((root/'config.json').read_text())!=cfg:
        raise ValueError('Use a new run directory for changed configurations')
    if args.command not in ('generate','all'):
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
        scenes=json.loads((root/'scenes.json').read_text()); report={}; env=ObstacleReach2D(cfg['env'])
        value=load_model(root,'value',cfg)
        try:
            for name in ['bc','sft','recap']:
                model=load_model(root,name,cfg); report[name]={}
                for split in (name for name in scenes if name not in ('train','validation') and scenes[name]):
                    with tqdm(scenes[split],desc=f'Evaluate {name}/{split}',unit='episode',dynamic_ncols=True) as progress:
                        eps=[rollout(env,s,policy(model),cfg['seed']+20000000+i,name) for i,s in enumerate(progress)]
                    annotate(eps,value,cfg)
                    for ep in eps: store(root,split,name,ep)
                    successes=[bool(e['success'][-1]) for e in eps]; n=len(eps); rate=float(np.mean(successes))
                    center=(rate+1.96**2/(2*n))/(1+1.96**2/n); half=1.96*np.sqrt(rate*(1-rate)/n+1.96**2/(4*n*n))/(1+1.96**2/n)
                    report[name][split]=dict(success_rate=rate,success_ci95=[center-half,center+half],
                        collision_rate=float(np.mean(['collision' in str(e['failure_reason'][-1]) for e in eps])),
                        timeout_rate=float(np.mean([e['truncated'][-1] for e in eps])),
                        completion_steps=float(np.mean([len(e['reward']) for e in eps if e['success'][-1]])) if any(successes) else None,
                        value_mae=float(np.mean(np.abs(np.concatenate([e['value_prediction']-e['mc_return'] for e in eps])))),
                        scene_ids=[str(e['scene_id']) for e in eps],successes=successes)
            for split in report['bc']:
                report['recap'][split]['gain_over_bc']=report['recap'][split]['success_rate']-report['bc'][split]['success_rate']
            save_json(root/'evaluation.json',report)
        finally: env.close()
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
    elif args.command=='train': fit(root,args.model,cfg,args.resume)
    elif args.command=='rollout': collect()
    elif args.command=='evaluate': evaluate()
    else: plot()

if __name__=='__main__': main()
