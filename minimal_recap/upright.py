"""Physical side grasp, wrist rotation and unsupported upright placement in x-z."""
import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from .env import ManipulationEnv
from .gripper import GripperMotion


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


class UprightEnv(ManipulationEnv):
    def __init__(self, config, budget=None, failure_cost=None):
        if config['environment']['lock_rotation']:
            raise ValueError('Upright placement requires object rotation')
        super().__init__(config, budget, failure_cost)
        self.action_space = spaces.Box(-1., 1., (4,), dtype=np.float32)

    def xml(self):
        c, s = self.c, self.scene
        hx, hy, hz = c['block_length']/2, c['block_width']/2, c['block_thickness']/2
        f, L = c['finger_thickness']/2, c['finger_length']/2
        return f'''<mujoco model="pedestal_upright"><compiler angle="radian"/>
<option timestep="{c['timestep']}" gravity="{' '.join(map(str,c['gravity']))}" integrator="implicitfast"/>
<visual><headlight ambient="0.5 0.5 0.5" diffuse="0.7 0.7 0.7"/></visual>
<default><geom friction="{c['friction']} 0.005 0.0001" solref="0.006 1" solimp="0.95 0.99 0.001"/><joint damping="0.1"/></default>
<worldbody><light pos="0 -0.5 1"/><light pos="0.5 0.5 0.8"/>
<camera name="front" pos="0 -0.95 0.45" xyaxes="1 0 0 0 0.23 1"/>
<geom name="table" type="plane" size="0.6 0.3 0.02" rgba="0.45 0.48 0.52 1"/>
<geom name="pedestal" type="box" pos="{c['pedestal_edge']-c['pedestal_width']/2} 0 {s['pedestal_height']/2}" size="{c['pedestal_width']/2} {c['pedestal_depth']/2} {s['pedestal_height']/2}" rgba="0.32 0.45 0.57 1"/>
<geom name="target" type="box" pos="{s['goal_x']} 0 0.0002" size="{c['target_half_width']} 0.04 0.0002" contype="0" conaffinity="0" rgba="0.2 0.85 0.3 0.6"/>
<body name="block" pos="{s['object_x']} 0 {s['pedestal_height']+hz}">
<joint name="ox" type="slide" axis="1 0 0"/><joint name="oz" type="slide" axis="0 0 1"/><joint name="angle" type="hinge" axis="0 1 0" damping="0.0001"/>
<geom name="object" type="box" size="{hx} {hy} {hz}" mass="{c['block_mass']}" rgba="0.95 0.48 0.12 1"/></body>
<body name="gripper">
<geom name="palm" type="box" pos="0 0 {L+0.009}" size="0.055 0.022 0.008" mass="0.15" rgba="0.25 0.28 0.33 1"/>
<body pos="{-f} 0 0"><joint name="left" type="slide" axis="-1 0 0" range="0 {c['max_opening']/2}" limited="true"/><geom name="lf" type="box" size="{f} 0.025 {L}" mass="0.03" rgba="0.18 0.3 0.85 1"/></body>
<body pos="{f} 0 0"><joint name="right" type="slide" axis="1 0 0" range="0 {c['max_opening']/2}" limited="true"/><geom name="rf" type="box" size="{f} 0.025 {L}" mass="0.03" rgba="0.18 0.3 0.85 1"/></body>
</body></worldbody>
<equality><joint joint1="left" joint2="right" polycoef="0 1 0 0 0" solref="0.004 1"/></equality>
<actuator><position joint="left" kp="{c['grip_kp']}" kv="{c.get('grip_kv',3)}" forcerange="-{c['grip_force']} {c['grip_force']}"/>
<position joint="right" kp="{c['grip_kp']}" kv="{c.get('grip_kv',3)}" forcerange="-{c['grip_force']} {c['grip_force']}"/>
</actuator></mujoco>'''

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        self.seed = seed
        c = self.c
        overhang = float(self.np_random.uniform(*c['overhang']))
        self.scene = {'task': 'pedestal_upright', 'layout': 'edge_to_table', 'overhang': overhang,
                      'pedestal_height': float(self.np_random.uniform(*c['pedestal_height'])),
                      'object_x': c['pedestal_edge']+overhang-c['block_length']/2,
                      'goal_x': float(self.np_random.uniform(*c['goal_x']))}
        if options and 'scene' in options:
            self.scene = copy.deepcopy(options['scene'])
        if self.renderer:
            self.renderer.close()
            self.renderer = None
        self.model = mujoco.MjModel.from_xml_string(self.xml())
        self.data = mujoco.MjData(self.model)
        self.initialize_robot()
        self.gripper_motion = GripperMotion(c.get('gripper_motion'), self.opening, c['max_opening'], c['timestep'])
        mujoco.mj_forward(self.model, self.data)
        self.steps = self.hold = self.grasps = self.drops = self.close_attempts = 0
        self.ever_grasped = self.was_grasped = self.first_grasp = self.previous_close = self.done = False
        return self.obs(), self.info('running')

    def initialize_robot(self):
        raise NotImplementedError("An articulated robot implementation is required")

    @property
    def grip_velocity(self):
        raise NotImplementedError("An articulated robot implementation is required")

    @property
    def wrist_velocity(self):
        raise NotImplementedError("An articulated robot implementation is required")

    def extra_observations(self):
        return {}

    def apply_control(self, target, theta, opening):
        raise NotImplementedError("An articulated robot implementation is required")

    def set_gripper_target(self, opening):
        self.gripper_motion.command(opening)
        self.data.ctrl[-2:] = self.gripper_motion.reference/2

    def advance_gripper(self):
        self.data.ctrl[-2:] = self.gripper_motion.advance()/2

    def advance_physics(self):
        for _ in range(self.substeps):
            self.advance_gripper()
            mujoco.mj_step(self.model,self.data)

    @property
    def wrist(self):
        raise NotImplementedError("An articulated robot implementation is required")

    def extent(self):
        c, a = self.c, self.angle()
        return np.array([abs(np.cos(a))*c['block_length']/2+abs(np.sin(a))*c['block_thickness']/2,
                         abs(np.sin(a))*c['block_length']/2+abs(np.cos(a))*c['block_thickness']/2])

    def angle_error(self):
        # Identical cuboid ends: upright orientation is equivalent modulo pi.
        diff = self.angle()-self.c['target_angle']
        return 0.5*np.arctan2(np.sin(2*diff), np.cos(2*diff))

    def supported(self):
        return self.contact('object','table') or self.contact('object','pedestal')

    def obs(self):
        c, v, a = self.c, self.data.qvel, self.angle()
        goal = np.array([self.scene['goal_x'],c['block_length']/2])
        groups = {
            'gripper': [*self.grip, *self.grip_velocity, np.sin(self.wrist), np.cos(self.wrist), self.wrist_velocity,
                        self.opening, v[self.v['left']]+v[self.v['right']]],
            'object': [*self.block,v[self.v['ox']],v[self.v['oz']],np.sin(a),np.cos(a),v[self.v['angle']]],
            'goal': [*goal,np.sin(c['target_angle']),np.cos(c['target_angle'])],
            'pedestal': [c['pedestal_edge'],c['pedestal_width'],self.scene['pedestal_height'],self.scene['overhang'],c['pedestal_depth']],
            'relative': [*(self.grip-self.block),*(self.block-goal),np.sin(2*self.angle_error()),np.cos(2*self.angle_error())],
            'geometry': [c['block_length'],c['block_width'],c['block_thickness']]}
        groups.update(self.extra_observations())
        return np.concatenate([np.asarray(groups[k]) for k in self.cfg['observation']['features']]).astype(np.float32)

    def success_conditions(self):
        c = self.c
        return bool(abs(self.block[0]-self.scene['goal_x']) < c['success_position_tolerance']
                    and abs(self.block[0]-self.scene['goal_x'])+self.extent()[0] <= c['target_half_width']
                    and abs(self.angle_error()) < c['success_angle_tolerance']
                    and self.contact('object','table') and not self.contact('object','pedestal')
                    and abs(self.block[1]-self.extent()[1]) < c['table_height_tolerance']
                    and np.linalg.norm(self.data.qvel[[self.v['ox'],self.v['oz']]]) < c['success_speed']
                    and abs(self.data.qvel[self.v['angle']]) < c['success_angular_speed']
                    and self.opening > c['block_thickness']+0.008
                    and np.linalg.norm(self.grip-self.block) > c['release_clearance']
                    and all(not self.contact('object',g) for g in ['lf','rf','palm']))

    def step(self, action):
        if self.done:
            raise RuntimeError('reset required after task termination')
        action = np.asarray(action,dtype=float)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError('Expected finite action [dx,dz,dtheta,opening]')
        self.executed_action = np.clip(action,-1,1).astype(np.float32)
        a, c = self.executed_action, self.c
        target = self.grip+a[:2]*c['max_delta']
        target[0] = np.clip(target[0],*c['workspace_x'])
        target[1] = np.clip(target[1],c['min_height'],c['max_height'])
        theta = np.clip(self.wrist+a[2]*c['max_angle_delta'],*c['wrist_range'])
        opening = (a[3]+1)*c['max_opening']/2
        self.apply_control(target, theta, opening)
        closing = a[3] < -0.5
        if closing and not self.previous_close:
            self.close_attempts += 1
        self.previous_close = closing
        self.advance_physics()
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError('Nonfinite physics')
        self.steps += 1
        grasp = self.grasped() and not self.supported()
        if grasp and not self.was_grasped:
            self.grasps += 1
            if not self.ever_grasped:
                self.first_grasp = self.close_attempts == 1
            self.ever_grasped = True
        # Count lost grasps in mid-air; supporting contact and intentional release are not drops.
        if self.was_grasped and not self.grasped() and not self.supported() and closing:
            self.drops += 1
        self.was_grasped = grasp
        self.hold = self.hold+1 if self.success_conditions() else 0
        success = self.hold >= c['success_hold_steps']
        failure = not c['workspace_x'][0] <= self.block[0] <= c['workspace_x'][1] or self.block[1] < -0.03
        timeout = self.budget is not None and self.steps >= self.budget
        reason = 'success' if success else 'out_of_bounds' if failure else 'task_timeout' if timeout else 'running'
        self.done = bool(success or failure or timeout)
        reward = 0. if success else (-float(self.failure_cost) if self.failure_cost is not None else np.nan) if self.done else -1.
        return self.obs(),reward,self.done,False,self.info(reason)

    def info(self, reason):
        result = super().info(reason)
        result.update(task='pedestal_upright', tilt_error=float(abs(self.angle_error())),
                      position_error=float(abs(self.block[0]-self.scene['goal_x'])),
                      recovered=reason=='success' and (self.close_attempts>1 or self.drops>0))
        result['gripper_motion'] = self.gripper_motion.snapshot()
        return result

    def snapshot(self):
        state = super().snapshot()
        state['gripper_motion'] = self.gripper_motion.snapshot()
        return state

    def restore(self, state):
        super().restore(state)
        if 'gripper_motion' in state:
            self.gripper_motion.restore(state['gripper_motion'])
        elif self.gripper_motion.enabled:
            raise ValueError('Smooth gripper resume requires its saved controller state')
        return self.obs()


class UprightExpert:
    """Feedback-only control; no qpos edits, welds, or attachment forces."""
    def __init__(self, env):
        self.env = env
        self.stage, self.ticks, self.attempts = 'open',0,0
        self.probe_start = None
        self.anchor = None
        self.mode = 'side'

    def state_dict(self):
        return {k: copy.deepcopy(getattr(self,k)) for k in ['stage','ticks','attempts','probe_start','anchor','mode']}

    def load_state_dict(self,state):
        for k,v in state.items():
            setattr(self,k,copy.deepcopy(v))

    def switch(self,stage):
        self.stage,self.ticks = stage,0

    def command(self,target,theta,opening):
        e = self.env
        return np.array([*np.clip((np.asarray(target)-e.grip)/e.c['max_delta'],-1,1),
                         np.clip(wrap(theta-e.wrist)/e.c['max_angle_delta'],-1,1),
                         2*opening/e.c['max_opening']-1],dtype=np.float32)

    def placement_wrist(self):
        e = self.env
        candidates = [e.wrist+wrap(e.c['target_angle']+k*np.pi-e.angle()) for k in [-1,0,1]]
        feasible = [x for x in candidates if e.c['wrist_range'][0] <= x <= e.c['wrist_range'][1]]
        if not feasible:
            raise RuntimeError('No upright orientation reachable from this grasp within wrist limits')
        return min(feasible, key=lambda x: abs(x-e.wrist))

    def __call__(self,obs):
        e,c = self.env,self.env.c
        cfg = e.cfg['expert_rollout']
        p,b = e.grip,e.block
        self.ticks += 1
        if self.ticks > cfg['stage_guard_steps']:
            raise RuntimeError(f'Upright expert stalled in {self.stage}; save as aborted for inspection')
        safe = e.scene['pedestal_height']+c['block_length']+cfg['rotation_clearance']
        if safe > c['max_height']-0.02:
            raise ValueError('Insufficient rotation clearance')
        tol = cfg['position_tolerance']
        target,theta,opening = p.copy(),e.wrist,c['max_opening']
        # After an actual fall to the table, use a top grasp on the short dimension.
        if self.stage == 'open':
            self.mode = 'side' if e.contact('object','pedestal') or b[1] > e.scene['pedestal_height'] else 'top'
            target = np.array([p[0],max(p[1],b[1]+cfg['reacquire_height'])]) if self.attempts else p
            if e.opening > c['max_opening']*0.9 and (not self.attempts or abs(p[1]-target[1])<tol):
                self.switch('align')
        elif self.stage in ['align','approach']:
            if self.mode == 'side':
                theta = np.pi/2+e.angle()
                point = b+np.array([np.cos(e.angle()),-np.sin(e.angle())])*(c['block_length']/2-cfg['grasp_depth'])
                target = point+np.array([c['initial_approach_gap'],0]) if self.stage=='align' else point
            else:
                # Nearest equivalent top-grasp wrist for a symmetric cuboid.
                theta = float(0.5*np.arctan2(np.sin(2*(e.angle()+np.pi/2)),np.cos(2*(e.angle()+np.pi/2))))
                target = np.array([b[0],b[1]+cfg['reacquire_height']]) if self.stage=='align' else np.array([b[0],max(c['finger_length']/2+0.005,b[1])])
            if np.max(abs(p-target))<tol and abs(wrap(theta-e.wrist))<cfg['angle_tolerance']:
                if self.stage=='align':
                    self.switch('approach')
                else:
                    self.attempts += 1
                    self.anchor = [float(x) for x in p]
                    self.switch('close')
        elif self.stage=='close':
            target,opening = np.array(self.anchor),0.
            ready = True
            if e.gripper_motion.enabled:
                gap_speed = abs(e.data.qvel[e.v['left']]+e.data.qvel[e.v['right']])
                ready = gap_speed < cfg.get('grip_settle_speed',0.005) and (e.grasped() or e.gripper_motion.reference < 0.001)
            if self.ticks >= cfg['close_steps'] and ready:
                self.probe_start = float(b[1])
                self.anchor = [float(p[0]),float(p[1]+cfg['lift_probe'])]
                self.switch('probe')
        elif self.stage=='probe':
            target,opening = np.array(self.anchor),0.
            if self.ticks >= cfg['probe_steps']:
                self.switch('lift' if e.grasped() and b[1]>self.probe_start+cfg['lift_probe']/2 else 'open')
        elif self.stage=='lift':
            target,opening = np.array([p[0],safe]),0.
            if not e.grasped():
                self.switch('open')
            elif abs(p[1]-safe)<tol:
                self.anchor = [float(x) for x in p]
                self.switch('rotate')
        elif self.stage=='rotate':
            target,opening = np.array(self.anchor),0.
            theta = self.placement_wrist()
            if not e.grasped():
                self.switch('open')
            elif abs(e.angle_error())<cfg['angle_tolerance'] and abs(e.data.qvel[e.v['angle']])<c['success_angular_speed']:
                self.switch('transport')
        elif self.stage in ['transport','lower']:
            opening = 0.
            theta = self.placement_wrist()
            target = p+np.array([e.scene['goal_x']-b[0],0.])
            target[1] = safe if self.stage=='transport' else p[1]+c['block_length']/2-b[1]
            if self.stage=='transport' and not e.grasped():
                self.switch('open')
            elif self.stage=='transport' and abs(b[0]-e.scene['goal_x'])<tol:
                self.switch('lower')
            elif self.stage=='lower' and e.contact('object','table') and abs(e.angle_error())<cfg['angle_tolerance']:
                self.anchor = [float(x) for x in p]
                self.switch('release')
        elif self.stage=='release':
            target = np.array(self.anchor)
            if self.ticks >= cfg['release_steps'] and e.opening>c['max_opening']*0.9:
                self.switch('wait')
        elif self.stage=='wait':
            target = np.array([p[0],max(cfg['retreat_height'],c['block_length']+c['release_clearance'])])
            if self.ticks > cfg['wait_steps'] and (abs(e.angle_error())>c['success_angle_tolerance'] or abs(b[0]-e.scene['goal_x'])>c['success_position_tolerance']):
                self.switch('open')
        return self.command(target,theta,opening)
