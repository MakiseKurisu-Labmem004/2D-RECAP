import copy
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from recap.env import ObstacleReach2D,clearance,points
from recap.models import returns,advantage,distribution_targets,Model
from recap.planning import ik,plan,ScenarioGenerator

@pytest.fixture
def cfg():
    return yaml.safe_load(Path('configs/default.yaml').read_text())

def test_returns_and_terminal_advantage():
    r=np.array([-1.,-1.,-150.])
    np.testing.assert_allclose(returns(r,349),[-152/349,-151/349,-150/349])
    np.testing.assert_allclose(advantage(r,np.zeros(3),np.ones(3)*99,10,349),returns(r,349))
    assert advantage(r,np.zeros(3),np.ones(3),1,349)[0]==pytest.approx(1-1/349)

def test_distribution_interpolation():
    y=torch.tensor([-1.,-.333,0.]); target=distribution_targets(y,201)
    torch.testing.assert_close(target.sum(-1),torch.ones(3))
    torch.testing.assert_close(target@torch.linspace(-1,0,201),y)

def test_geometry_ik():
    scene=dict(obstacle=[.25,0],radius=.05)
    assert clearance([0,0],scene,.015)[0]<0
    scene['obstacle']=[.75,0]
    assert clearance([0,0],scene,.015)[1]<0
    for q in ik(np.array([.5,.5]),3.): np.testing.assert_allclose(points(q)[-1],[.5,.5],atol=1e-7)

def test_environment_timeout_and_action(cfg):
    c=copy.deepcopy(cfg['env']); c['horizon']=1
    env=ObstacleReach2D(c)
    try:
        o,_=env.reset(seed=1)
        assert o.shape==(9,) and env.observation_space.contains(o)
        env.data.qvel[:]=[2.,-3.]
        np.testing.assert_array_equal(env.obs(),o)
        env.data.qvel[:]=0
        q=env.data.qpos.copy()
        _,r,term,trunc,info=env.step(np.zeros(2))
        np.testing.assert_allclose(info['applied_command'],q)
        assert trunc and not term and r==-150
        with pytest.raises(RuntimeError): env.step(np.zeros(2))
    finally: env.close()

def test_collision(cfg):
    env=ObstacleReach2D(cfg['env'])
    try:
        env.reset(options={'scene':dict(q0=[0.,0.],goal=[0.,.8],obstacle=[.25,0.],radius=.07)})
        _,r,term,trunc,info=env.step(np.zeros(2))
        assert term and not trunc and r==-150 and info['collision']
    finally: env.close()

def test_models(cfg):
    x=torch.randn(8,9)
    for kind in ['bc','recap','value']:
        m=Model(kind,cfg['train'],.05)
        if kind=='value':
            assert m(x).shape==(8,201)
            assert torch.all((m.value(x)>=-1)&(m.value(x)<=0))
        else: assert m.action(x).abs().max()<=.05
    loss=m.value(x).sum(); loss.backward()

def test_executable_scenario(cfg):
    cfg.pop('difficulty_ratio',None)
    cfg['planner']['max_attempts']=100
    pairs=ScenarioGenerator(cfg).generate('train',1,42)
    assert len(pairs)==1 and pairs[0][1]['success'][-1]
    assert len(pairs[0][1]['reward'])<=100
