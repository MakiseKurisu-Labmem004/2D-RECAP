import copy
import numpy as np
import mujoco
import pytest
from minimal_recap.config import load
from minimal_recap.gripper import GripperMotion
from minimal_recap.tasks import make_env


def test_reference_limits_during_full_strokes_and_rapid_reversals():
    motion = GripperMotion(dict(max_speed=.06, max_acceleration=.4, response_time=.2), .08, .08, .0025)
    rng = np.random.default_rng(19)
    commands = [0.]*1000 + [.08]*1000 + list(rng.uniform(0,.08,3000)) + [0.]*1000
    for target in commands:
        x,v = motion.reference,motion.velocity
        motion.command(target)
        motion.advance()
        assert 0 <= motion.reference <= .08
        assert abs(motion.velocity) <= .06+1e-12
        assert abs(motion.reference-x) <= .06*.0025+1e-12
        assert abs(motion.velocity-v) <= .4*.0025+1e-12
    assert motion.reference < 1e-5
    with pytest.raises(ValueError, match='response_time'):
        GripperMotion(dict(max_speed=.06,max_acceleration=.4,response_time=.01),0,.08,.0025)


def test_physical_gripper_speed_and_mid_ramp_resume(monkeypatch):
    c=load('configs/experiments/arm3r.yaml')
    e=make_env(c)
    real_step=mujoco.mj_step
    velocities=[]
    def track(model,data):
        real_step(model,data)
        velocities.append(abs(data.qvel[e.v['left']]+data.qvel[e.v['right']]))
    monkeypatch.setattr(mujoco,'mj_step',track)
    try:
        e.reset(seed=10000)
        for _ in range(3):
            e.step([0,0,0,-1])
        state=e.snapshot()
        assert state['gripper_motion']['velocity'] < 0
        first=e.step([0,0,0,1])
        after=e.snapshot()
        e.restore(state)
        second=e.step([0,0,0,1])
        np.testing.assert_allclose(first[0],second[0],atol=1e-8)
        np.testing.assert_allclose(after['physics'],e.snapshot()['physics'],atol=1e-8)
        assert after['gripper_motion']==e.snapshot()['gripper_motion']
        for opening in [-1]*40+[1]*40+[-1,1]*10:
            e.step([0,0,0,opening])
            assert np.all(abs(e.data.actuator_force[-2:]) <= e.c['grip_force']+1e-8)
        # Dynamic response can slightly exceed reference speed; never clamp qvel.
        assert max(velocities) < .075
        assert e.opening > .02
        assert e.obs().shape == (40,)
        # Missing hidden controller state must not silently resume a different motion.
        invalid=copy.deepcopy(state)
        del invalid['gripper_motion']
        with pytest.raises(ValueError,match='saved controller state'):
            e.restore(invalid)
    finally:
        e.close()
