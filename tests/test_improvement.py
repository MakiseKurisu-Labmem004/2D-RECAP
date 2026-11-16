import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from recap import cli
from recap import improvement as improve
from recap.env import ObstacleReach2D
from recap.geometry import points
from recap.models import Model, warm_start_policy
from recap.planning import rollout


@pytest.fixture
def cfg():
    cfg=yaml.safe_load(Path('configs/default.yaml').read_text())
    cfg['device']='cpu'
    cfg['env'].update(horizon=60,hold=2)
    cfg['data']['exploration']=0.
    cfg['train'].update(width=8,depth=2,max_steps=2,goal_anchor_samples=0,validation_steps=1,checkpoint_steps=1)
    return cfg


@pytest.mark.parametrize('kind',['bc','sft'])
def test_gaussian_warm_start_preserves_source_actions_for_all_conditions(cfg,kind):
    source=Model(kind,cfg['train'],cfg['env']['action_limit'])
    target=Model('recap',cfg['train'],cfg['env']['action_limit'])
    warm_start_policy(target,source)
    obs=torch.randn(20,9)
    for condition in (0,1,2):
        logits=target(obs,torch.full((len(obs),),condition,dtype=torch.long))[:,:2]
        torch.testing.assert_close(logits.tanh()*target.limit,source.action(obs),atol=1e-7,rtol=1e-5)
    next_policy=Model('recap',cfg['train'],target.limit)
    warm_start_policy(next_policy,target)
    torch.testing.assert_close(next_policy.action(obs),target.action(obs),atol=0,rtol=0)


def test_fixed_target_holdout_is_disjoint_and_deterministic():
    scenes=[dict(scene_id=str(i),split='id',metrics=dict(difficulty='hard' if i<5 else 'easy')) for i in range(20)]
    a,h=improve.partition_scenes(scenes,.2,42)
    assert len(a)==16 and len(h)==4
    assert {s['scene_id'] for s in a}.isdisjoint(s['scene_id'] for s in h)
    assert (a,h)==improve.partition_scenes(scenes,.2,42)


def test_unified_pool_merges_sources_without_modifying_manifest():
    manifest=dict(train=[dict(scene_id='t')],validation=[dict(scene_id='v')],
                  id=[dict(scene_id='i')],outer_goal_ood=[dict(scene_id='o')])
    before=copy.deepcopy(manifest)
    pool,counts=improve.select_scene_pool(manifest,['all'])
    assert [s['scene_id'] for s in pool]==['t','i','o']
    assert counts==dict(train=1,id=1,outer_goal_ood=1)
    practice,holdout=improve.partition_scenes(pool,0,42)
    assert practice==pool and holdout==[]
    assert manifest==before
    with pytest.raises(ValueError,match='reserved for monitoring'):
        improve.select_scene_pool(manifest,['validation'])


def test_default_command_pools_all_scenes_without_holdout(tmp_path,monkeypatch):
    captured=[]
    monkeypatch.setattr(improve,'run_experiment',lambda source,output,modes,splits,options:captured.append((splits,options)))
    improve.main(['--run',str(tmp_path)])
    splits,options=captured[0]
    assert splits==['all'] and options.holdout_fraction==0


def test_regression_changes_only_reports_metrics():
    before=dict(a=True,b=False)
    good=improve.regression_changes(before,dict(a=True,b=True),dict(v=True),dict(v=True))
    assert good['recovered']==['b'] and 'accepted' not in good
    bad=improve.regression_changes(before,dict(a=False,b=True),dict(v=True),dict(v=False))
    assert bad['regressions']==['a'] and bad['validation_regressions']==['v']


def test_takeover_preserves_physics_and_continues_until_success_beyond_deadline(cfg,monkeypatch):
    scene=dict(q0=[-.6,1.],goal=points([.3,1.])[-1].tolist(),obstacle=[-.7,.4],radius=.05,scene_id='x')
    env=ObstacleReach2D(cfg['env'])
    try: failed=rollout(env,scene,lambda obs,env:np.array([.005,0.]),42,'practice',limit=30)
    finally:env.close()
    cut=27
    reference=ObstacleReach2D(cfg['env'])
    try:
        reference.reset(seed=42,options={'scene':scene})
        for action in failed['action'][:cut]: reference.step(action)
        expected_qvel=reference.data.qvel.copy()
        expected_filter=reference.command_filter.state.copy()
    finally:reference.close()
    assert np.linalg.norm(expected_qvel)>1e-5
    checked=[]
    def expert(obs,env):
        if not checked:
            np.testing.assert_allclose(env.data.qvel,expected_qvel,atol=1e-12)
            np.testing.assert_allclose(env.command_filter.state,expected_filter,atol=1e-12)
            checked.append(True)
        return .5*(np.array([.3,1.])-env.data.qpos)
    monkeypatch.setattr(improve,'plan',lambda start,goal,scene,cfg:np.array([start,goal]))
    monkeypatch.setattr(improve,'make_expert',lambda *args:expert)
    corrected,report=improve.takeover_episode(scene,failed,cfg,123,3)
    assert checked and report['expert_steps']>50 and report['takeover_step']==cut
    assert corrected['intervention_mask'].all() and corrected['success'][-1]
    assert report['policy_suffix_steps']==0 and not bool(corrected['task_success'])
    np.testing.assert_array_equal(corrected['observation'][0],failed['observation'][cut])
    assert corrected['observation'].shape[1]==9


@pytest.mark.parametrize('splits,fraction,pool_size,holdout_size',[(['id'],.5,4,2),(['all'],0.,5,0)])
def test_pair_keeps_original_data_and_limits_extra_expert_labels(tmp_path,cfg,monkeypatch,splits,fraction,pool_size,holdout_size):
    source=tmp_path/'run';source.mkdir()
    manifest=dict(train=[dict(scene_id='train')],validation=[dict(scene_id='validation')],
                  id=[dict(scene_id=f'id{i}',split='id',metrics=dict(difficulty='easy')) for i in range(4)])
    (source/'config.json').write_text(json.dumps(cfg));(source/'scenes.json').write_text(json.dumps(manifest))
    torch.save(dict(config=cfg,model=Model('bc',cfg['train'],.04).state_dict()),source/'bc.pt')
    before={p:p.read_bytes() for p in source.iterdir()}
    calls=[];collected=[]
    baseline=SimpleNamespace(name='baseline')
    candidate=SimpleNamespace(name='candidate')
    monkeypatch.setattr(cli,'load_model',lambda root,kind,cfg,checkpoint=None:baseline if root==source else candidate)
    def evaluate(model,scenes,cfg,seed,description,**kwargs):
        return dict(metrics={},successes={s['scene_id']:model.name=='baseline' for s in scenes})
    monkeypatch.setattr(improve,'evaluate_scenes',evaluate)
    def episode(scene,seed):
        return dict(observation=np.zeros((2,9)),action=np.zeros((2,2)),reward=-np.ones(2),
                    intervention_mask=np.zeros(2,dtype=bool),episode_id=np.array(f'e{seed}'),seed=np.array(seed),
                    scene_id=np.array(scene['scene_id']),distance=np.ones(2),success=np.zeros(2,dtype=bool),
                    policy_checkpoint_sha256=np.array('hash'))
    def collect(model,scenes,cfg,count,seed,root):
        collected.append([s['scene_id'] for s in scenes])
        pairs=[(scenes[i%len(scenes)],episode(scenes[i%len(scenes)],seed+i)) for i in range(count)]
        for _,ep in pairs:cli.store(root,'adaptation','rollout',ep)
        return [ep for _,ep in pairs],pairs
    monkeypatch.setattr(improve,'collect',collect)
    def takeover(scene,failed,cfg,seed,lookback):
        ep=episode(scene,seed)
        ep.update(observation=np.zeros((3,9)),reward=-np.ones(3),distance=np.ones(3),action=np.zeros((3,2)),intervention_mask=np.ones(3,dtype=bool),
                  success=np.array([False,False,True]),task_success=np.array(True),takeover_step=np.array(1))
        return ep,dict(status='collected',expert_steps=3,replay_steps=1,policy_suffix_steps=0,success=True,task_success=True)
    monkeypatch.setattr(improve,'takeover_episode',takeover)
    def fit(root,kind,cfg,**kwargs):
        calls.append((root.parent.name,kind,kwargs['expert_root'],copy.deepcopy(kwargs['extra_train'])))
        if kind=='value':(root/'value.pt').write_bytes(b'critic snapshot')
        if kind=='recap':
            for ep in kwargs['extra_train']:
                n=len(ep['action'])
                ep.update(value_prediction=np.zeros(n),mc_return=-np.ones(n),td_advantage=-np.ones(n),
                          advantage_indicator=ep['intervention_mask'].copy())
    monkeypatch.setattr(cli,'fit',fit)
    args=SimpleNamespace(device='cpu',seed=42,noise=0.,steps=2,lr=.0001,correction_weight=5.,train_config=None,
                         scene_ids=None,holdout_fraction=fraction,rounds=2,episodes=4,expert_episodes=3,lookback=2)
    output=source/'improvement'/'test'
    result=improve.run_experiment(source,output,['autonomous','corrections'],splits,args)
    protocol=json.loads((output/'protocol.json').read_text())
    assert (output/'initial.bc.pt').read_bytes()==(source/'bc.pt').read_bytes()
    holdout=set(protocol['holdout_scene_ids'])
    assert len(holdout)==holdout_size and protocol['scene_pool_size']==pool_size
    assert 'validation' not in protocol['scene_pool_sources']
    assert all(not holdout.intersection(group) for group in collected)
    assert all(expert_root==source for _,_,expert_root,_ in calls)
    pure=[extra for mode,_,_,extra in calls if mode=='autonomous']
    assert all(not ep['intervention_mask'].any() for extra in pure for ep in extra)
    for mode,kind,_,extra in calls:
        assert all(str(ep['scene_id']) not in holdout for ep in extra)
    budget=result['corrections']['rounds'][0]['budget']
    assert budget['expert_steps']==9 and budget['expert_attempts']==3
    assert budget['expert_recovery_successes']==3 and budget['assisted_successes']==3
    assert result['autonomous']['rounds'][0]['budget']['expert_steps']==0
    assert result['autonomous']['rounds'][0]['budget']['autonomous_episodes']==budget['autonomous_episodes']==4
    for arm in result.values():
        assert arm['rounds'][0]['changes']['regressions']
        assert all(r['policy_updated'] and 'gate' not in r for r in arm['rounds'])
        assert arm['active_policy_kind']=='recap' and 'round-002' in arm['active_policy']
    annotations=output/'corrections/round-002/annotations.json'
    index=json.loads(annotations.read_text())
    assert len(index['trajectories'])==14
    for row in index['trajectories']:
        sidecar=annotations.parent/row['annotations']
        with np.load(sidecar,allow_pickle=False) as recorded:
            assert len(recorded['td_advantage'])==row['frames']
            assert (sidecar.parent/str(recorded['trajectory_path'])).resolve().is_file()
            raw_path=(sidecar.parent/str(recorded['trajectory_path'])).resolve()
            with np.load(raw_path,allow_pickle=False) as raw:
                assert len(raw['td_advantage'])==row['frames']
                expected_round=1 if 'round-001' in str(raw_path) else 2
                assert int(raw['annotation_round'])==expected_round
    for p,content in before.items():assert p.read_bytes()==content


def test_demonstrations_and_only_teacher_frames_receive_forced_positive_labels(tmp_path,cfg,monkeypatch):
    from recap.rewards import REWARD_VERSION
    cfg['train']['reward_version']=REWARD_VERSION
    expert=dict(observation=np.zeros((3,9),dtype=np.float32),action=np.zeros((3,2),dtype=np.float32),
                reward=-np.ones(3),success=np.array([True]),episode_id=np.array('original'),scene_id=np.array('same-scene'))
    corrected=copy.deepcopy(expert)
    corrected.update(episode_id=np.array('correction'),intervention_mask=np.array([False,True,False]))
    autonomous=copy.deepcopy(expert)
    autonomous.update(episode_id=np.array('autonomous'),action_source=np.array('policy'))
    monkeypatch.setattr(cli,'episodes',lambda root,split,source='expert':
                        [copy.deepcopy(expert)] if source=='expert' else [copy.deepcopy(autonomous)])
    def annotate(episodes,model,cfg):
        for ep in episodes:ep['td_advantage']=-np.ones(3)
    monkeypatch.setattr(cli,'annotate',annotate)
    torch.save(dict(config=cfg,model=Model('value',cfg['train'],.04).state_dict()),tmp_path/'value.pt')
    (tmp_path/'scenes.json').write_text(json.dumps(dict(train=[],validation=[])))
    captured=[]
    monkeypatch.setattr(cli,'train_steps',lambda *args:captured.append((args[5][2].numpy(),args[6][2].numpy())))
    cli.fit(tmp_path,'recap',cfg,expert_root=tmp_path,extra_train=[corrected])
    labels,validation_labels=captured[0]
    np.testing.assert_array_equal(labels[:3,0],[1,1,1])
    np.testing.assert_array_equal(labels[3:6,0],[0,0,0])
    np.testing.assert_array_equal(labels[6:,0],[0,1,0])
    np.testing.assert_array_equal(labels[:,1],[0,0,0,0,0,0,0,1,0])
    np.testing.assert_array_equal(validation_labels[:,0],[1,1,1,0,0,0])
    diagnostics=json.loads((tmp_path/'diagnostics.json').read_text())
    assert diagnostics['positive_fraction']==0 and diagnostics['estimated_selected_fraction']==0
    assert diagnostics['selected_fraction']==pytest.approx(4/9)
    assert diagnostics['expert_demonstration_steps']==3 and diagnostics['intervention_steps']==1
    with np.load(tmp_path/'data/train/annotated/original.npz') as stored:
        assert stored['advantage_indicator'].all() and bool(stored['expert_demonstration'])
        assert (stored['td_advantage']==-1).all()


def test_uniform_sampling_is_independent_of_success_or_failure(tmp_path,cfg,monkeypatch):
    scenes=[dict(scene_id='a'),dict(scene_id='b')]
    successful=[True]
    class Env:
        def __init__(self,cfg):pass
        def close(self):pass
    monkeypatch.setattr(improve,'ObstacleReach2D',Env)
    monkeypatch.setattr(cli,'store',lambda *args:None)
    def sample(env,scene,actor,seed,version,noise):
        return dict(action=np.zeros((1,2)),success=np.array([successful[0]]),
                    scene_id=np.array(scene['scene_id']),episode_id=np.array(str(seed)))
    monkeypatch.setattr(improve,'rollout',sample)
    first,failures=improve.collect(SimpleNamespace(),scenes,cfg,500,42,tmp_path)
    assert not failures
    successful[0]=False
    second,failures=improve.collect(SimpleNamespace(),scenes,cfg,500,42,tmp_path)
    ids=[str(ep['scene_id']) for ep in first]
    assert ids==[str(ep['scene_id']) for ep in second]
    assert len(failures)==500
    assert .4<ids.count('a')/len(ids)<.6


def test_value_targets_keep_task_timeout_for_extended_expert_recovery(cfg):
    ep=dict(takeover_step=np.array(58),action=np.zeros((5,2)),observation=np.zeros((5,9)),
            reward=np.array([-1.,-1.,-1.,-1.,0.]),terminated=np.array([False]*4+[True]),
            truncated=np.zeros(5,dtype=bool),success=np.array([False]*4+[True]),
            failure_reason=np.array(['']*5),intervention_mask=np.ones(5,dtype=bool),initial_qpos=np.array([1.,2.]))
    view=improve.correction_value_episode(ep,cfg)
    assert len(view['action'])==2 and view['truncated'][-1] and not view['terminated'][-1]
    assert not view['success'].any() and view['reward'][-1]==-cfg['env']['horizon'] and view['failure_reason'][-1]=='timeout'
    assert ep['success'][-1] and len(ep['action'])==5 and ep['reward'][-1]==0
    np.testing.assert_array_equal(view['initial_qpos'],[1.,2.])


def test_regular_rollout_keeps_deadline_while_teacher_can_finish(cfg):
    cfg['env'].update(horizon=1,hold=2)
    scene=dict(q0=[-.6,1.],goal=points([-.6,1.])[-1].tolist(),obstacle=[-.7,.4],radius=.05,scene_id='x')
    env=ObstacleReach2D(cfg['env'])
    try:
        regular=rollout(env,scene,lambda *args:np.zeros(2),42,'policy')
        teacher=rollout(env,scene,lambda *args:np.zeros(2),42,'teacher',until_done=True)
    finally:env.close()
    assert len(regular['action'])==1 and regular['truncated'][-1] and not regular['success'][-1]
    assert len(teacher['action'])==2 and teacher['success'][-1] and not teacher['truncated'][-1]


def test_stationary_expert_failure_is_not_recorded_as_success(cfg,monkeypatch):
    scene=dict(q0=[-.6,1.],goal=points([.3,1.])[-1].tolist(),obstacle=[-.7,.4],radius=.05,scene_id='x')
    env=ObstacleReach2D(cfg['env'])
    try:failed=rollout(env,scene,lambda *args:np.zeros(2),42,'practice')
    finally:env.close()
    class Stalled:
        time=0.
        trajectory=SimpleNamespace(duration=0.)
        def __call__(self,*args):return np.zeros(2)
    monkeypatch.setattr(improve,'plan',lambda start,goal,scene,cfg:np.array([start,goal]))
    monkeypatch.setattr(improve,'make_expert',lambda *args:Stalled())
    ep,result=improve.takeover_episode(scene,failed,cfg,123,3)
    assert ep is None and result['status']=='expert_stalled'


def test_round_evaluation_saves_trajectory_with_frame_aligned_advantage(tmp_path,cfg):
    cfg['env'].update(horizon=1,hold=2)
    scene=dict(q0=[-.6,1.],goal=points([-.6,1.])[-1].tolist(),obstacle=[-.7,.4],radius=.05,scene_id='x')
    model=Model('bc',cfg['train'],cfg['env']['action_limit'])
    value=Model('value',cfg['train'],cfg['env']['action_limit'])
    with torch.no_grad():
        for parameter in model.parameters():parameter.zero_()
    improve.evaluate_scenes(model,[scene],cfg,42,'test',record_root=tmp_path,value=value)
    files=list((tmp_path/'data/evaluation/practice').glob('*.npz'))
    assert len(files)==1
    with np.load(files[0],allow_pickle=False) as ep:
        assert len(ep['action'])==len(ep['td_advantage'])==len(ep['value_prediction'])==1
        assert ep['value_alignment'].item()=='observation_before_action'
        assert ep['mc_return'][0]==-1
        np.testing.assert_allclose(ep['td_advantage'],-1-ep['value_prediction'],atol=1e-7)
