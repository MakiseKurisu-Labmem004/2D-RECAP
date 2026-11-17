import mujoco
import numpy as np
import pytest
from minimal_recap.config import load
from minimal_recap.tasks import make_env, make_expert


@pytest.fixture
def placed():
    """Reach the table through real grasping, then deliberately keep the hand closed."""
    e=make_env(load('configs/experiments/arm3r.yaml'))
    try:
        obs,_=e.reset(seed=10000)
        expert=make_expert(e)
        for _ in range(500):
            action=expert(obs)
            if expert.stage=='release' and e.contact('object','table'):
                break
            obs,_,done,_,_=e.step(action)
            assert not done
        else:
            pytest.fail('Expert did not reach tabletop placement')
        assert e.grasped()
        for _ in range(60):
            _,reward,done,_,info=e.step([0,0,0,-1])
            if done:
                break
        assert done and info['success'] and reward==0
        yield e
    finally:
        e.close()


def test_success_allows_closed_nearby_gripper(placed):
    e=placed
    assert e.grasped()
    assert e.opening < 0.048
    assert np.linalg.norm(e.grip-e.block) < 0.1
    assert e.hold==e.c['success_hold_steps']
    assert e.success_conditions()


def test_goal_membership_uses_center_not_tight_alignment_or_footprint(placed):
    e=placed
    state=e.snapshot()
    state['scene']['goal_x']=float(e.block[0]+0.03)
    e.restore(state)
    assert abs(e.block[0]-e.scene['goal_x'])>0.012
    assert abs(e.block[0]-e.scene['goal_x'])+e.extent()[0]>e.c['target_half_width']
    assert e.success_conditions()
    e.scene['goal_x']=float(e.block[0]+0.041)
    assert not e.success_conditions()


def test_success_still_requires_upright_stationary_tabletop_state(placed):
    e=placed
    state=e.snapshot()
    for field,joint,delta in [('qpos','angle',0.2),('qvel','ox',0.03),
                              ('qvel','angle',0.3),('qpos','oz',0.04)]:
        e.restore(state)
        address=e.j[joint] if field=='qpos' else e.v[joint]
        getattr(e.data,field)[address]+=delta
        mujoco.mj_forward(e.model,e.data)
        assert not e.success_conditions(),(field,joint)


def test_stability_hold_restarts_after_leaving_goal(placed):
    e=placed
    e.done=False
    e.hold=10
    goal=e.scene['goal_x']
    e.scene['goal_x']=float(e.block[0]+0.06)
    _,_,done,_,_=e.step([0,0,0,-1])
    assert not done and e.hold==0
    e.scene['goal_x']=goal
    for i in range(e.c['success_hold_steps']):
        _,reward,done,_,info=e.step([0,0,0,-1])
        assert e.hold==i+1
        assert done==(i==e.c['success_hold_steps']-1)
    assert info['success'] and reward==0
