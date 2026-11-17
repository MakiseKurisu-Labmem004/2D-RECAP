import copy
import numpy as np
import pytest
import torch
from minimal_recap.config import load, merge
from minimal_recap.tasks import make_env, make_expert
from minimal_recap.learning import Network, Behavior, train, advantages
from minimal_recap.data import collect_episode, read_episode, freeze_contract
from minimal_recap.evaluation import make_recovery_pool


@pytest.fixture
def c():
    config = load('configs/experiments/smoke.yaml')
    config['model']['device'] = 'cpu'  # Unit tests do not require server hardware.
    return config


@pytest.mark.parametrize('height_index', [0,1])
@pytest.mark.parametrize('overhang_index', [0,1])
@pytest.mark.parametrize('goal_index', [0,1])
def test_side_lift_rotation_release_and_stability(c,height_index,overhang_index,goal_index):
    # Exercise all eight corners of the unified ID range, including combinations
    # that used to be excluded by separate evaluation overrides.
    ranges={key: [c['environment'][key][i]]*2 for key,i in
            [('pedestal_height',height_index),('overhang',overhang_index),('goal_x',goal_index)]}
    cfg=merge(c,{'environment':ranges})
    e=make_env(cfg)
    obs,_=e.reset(seed=10000)
    p=make_expert(e)
    lifted=False
    initial=e.angle()
    for _ in range(500):
        obs,r,done,trunc,info=e.step(p(obs))
        lifted |= e.grasped() and not e.supported()
        if done:
            break
    assert done and info['success'] and lifted
    assert e.steps>=c['environment']['success_hold_steps']
    assert abs(abs(e.angle()-initial)-np.pi/2)<0.1
    assert e.success_conditions() and e.contact('object','table')
    assert abs(e.angle_error())<c['environment']['success_angle_tolerance']
    assert e.hold==c['environment']['success_hold_steps']
    # Extra post-success observation, no physics edits; verify free-standing stability.
    for _ in range(40):
        e.done=False
        e.step([0,0,0,1])
    assert abs(e.angle_error())<c['environment']['success_angle_tolerance']
    e.close()


def test_four_action_scaling_and_exact_snapshot(c):
    e=make_env(c,300,300)
    obs,_=e.reset(seed=44)
    assert e.action_space.shape==(4,) and obs.shape==(34,)
    state=e.snapshot()
    a=[0.5,0.5,-0.5,-1]
    first=e.step(a)
    e.restore(state)
    second=e.step(a)
    np.testing.assert_allclose(first[0],second[0],atol=1e-8)
    with pytest.raises(ValueError):
        e.step([0,0,1])
    # Step counter does not appear in observations.
    e.restore(state)
    before=e.obs()
    e.steps=77
    np.testing.assert_array_equal(before,e.obs())
    # A quiet horizontal block on its pedestal is not a successful placement.
    assert not e.success_conditions()
    e.close()


def test_four_action_training_value_and_conditioned_update(c,tmp_path):
    c=merge(c,{k:{'epochs':1,'updates_per_epoch':1} for k in ['bc_training','value_training','recap']})
    eps=[]
    for i in range(2):
        e=make_env(c,300,300)
        p=tmp_path/f'e{i}.npz'
        collect_episode(e,make_expert(e),10000+i,{'episode_id':f'e{i}','source':'bc'},p)
        eps.append(read_episode(p))
        e.close()
    contract=freeze_contract(c,eps,tmp_path/'contract.json')
    bc=train(c,eps,tmp_path/'bc.pt','bc',contract)
    assert bc.net[0].in_features==34
    assert bc(torch.tensor(eps[0]['observation'][0])).shape==(4,)
    behavior=Behavior(bc,{'mode':'gaussian','std':[0.01]*4},1)
    assert behavior(eps[0]['observation'][0]).shape==(4,)
    value=train(c,eps,tmp_path/'value.pt','value',contract)
    assert value.net[0].in_features==34
    labels={}
    for ep in eps:
        vs=value.value(torch.tensor(ep['observation'])).numpy()
        ns=value.value(torch.tensor(ep['next_observation'])).numpy()
        labels[ep['metadata']['episode_id']]=advantages(ep,vs,ns,10)
    updated=train(c,eps,tmp_path/'updated.pt','update',contract,bc,eps,labels)
    assert updated.net[0].in_features==35
    assert updated.conditional and updated(torch.tensor(eps[0]['observation'][0])).shape==(4,)


def test_recovery_pool_preserves_remaining_budget(c,tmp_path):
    c['evaluation']['pool_size']=1
    states=make_recovery_pool(c,{'failure_cost':300},tmp_path/'pool.json')
    assert {s['mode'] for s in states}=={'empty','drop'}
    for item in states:
        e=make_env(c,300,300)
        e.restore(item['state'])
        assert e.budget-e.steps==item['remaining_budget']
        assert not e.done and not e.grasped()
        assert item['controller']['perturbation']['injected_steps']>0
        e.close()
