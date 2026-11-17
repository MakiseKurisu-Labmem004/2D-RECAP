import numpy as np
import mujoco
from minimal_recap.config import load
from minimal_recap.tasks import make_env,make_expert


def test_arm_joint_chain_limits_physical_grasp_and_release():
    c=load('configs/experiments/arm3r.yaml')
    env=make_env(c)
    obs,_=env.reset(seed=10000)
    assert obs.shape==(40,)
    assert env.action_space.shape==(4,)
    assert all(env.model.joint(n).type[0]==mujoco.mjtJoint.mjJNT_HINGE for n in ['shoulder','elbow','arm_wrist'])
    assert mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_JOINT,'gx')==-1
    initial=env.data.qpos[env.arm_q].copy()
    expert=make_expert(env)
    grasped=False
    for _ in range(400):
        obs,_,done,_,info=env.step(expert(obs))
        limits=np.asarray(c['environment']['arm_joint_ranges'])
        assert np.all(env.data.qpos[env.arm_q]>=limits[:,0]-1e-3)
        assert np.all(env.data.qpos[env.arm_q]<=limits[:,1]+1e-3)
        assert np.all(abs(env.data.actuator_force[:3])<=np.asarray(c['environment']['arm_torque'])+1e-7)
        grasped |= env.grasped() and not env.supported()
        if done:break
    assert done and info['success'] and grasped
    assert np.linalg.norm(env.data.qpos[env.arm_q]-initial)>0.3
    assert not env.grasped() and env.hold==20
    for _ in range(40):
        env.done=False
        env.step([0,0,0,1])
    assert abs(env.angle_error())<c['environment']['success_angle_tolerance']
    env.close()


def test_arm_ik_and_snapshot_replay():
    c=load('configs/experiments/arm3r.yaml');e=make_env(c)
    for point,theta in [([0.02,0.15],np.pi/2),([-0.06,0.29],0.),([0.17,0.12],0.)]:
        q=e.inverse_kinematics(point,theta)
        e.data.qpos[e.arm_q]=q
        mujoco.mj_forward(e.model,e.data)
        np.testing.assert_allclose(e.grip,point,atol=1e-10)
        assert abs(e.wrist-theta)<1e-10
    e.reset(seed=10000)
    e.step([0.2,0.3,-0.1,1])
    state=e.snapshot()
    first=e.step([0.1,0.1,-0.1,1])
    e.restore(state)
    second=e.step([0.1,0.1,-0.1,1])
    np.testing.assert_allclose(first[0],second[0],atol=1e-8)
    # Rejected poses must be projected or held, never teleport the joint state.
    before=e.data.qpos.copy()
    e.apply_control(np.array([5.,5.]),0.,0.08)
    np.testing.assert_array_equal(e.data.qpos,before)
    assert e.ik_scale<1
    e.close()
