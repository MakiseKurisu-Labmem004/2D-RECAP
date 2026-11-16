import copy
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces


class PlanarGraspEnv(gym.Env):
    """No TimeLimit wrapper. Task deadlines are terminal; external cuts are resumable."""
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 20}

    def __init__(self, config, budget=None, failure_cost=None):
        self.cfg = config
        self.c = config['environment']
        self.budget = budget
        self.failure_cost = failure_cost
        self.action_space = spaces.Box(-1., 1., (3,), dtype=np.float32)
        self.renderer = None
        dt = self.c['timestep']
        self.substeps = round(1 / self.c['control_hz'] / dt)
        if not np.isclose(self.substeps * dt * self.c['control_hz'], 1):
            raise ValueError('Control period must be an integer multiple of timestep')
        self.reset(seed=0)
        self.observation_space = spaces.Box(-np.inf, np.inf, self.obs().shape, dtype=np.float32)

    def xml(self):
        c, s = self.c, self.scene
        b = c['block_size'] / 2
        f = c['finger_thickness'] / 2
        L = c['finger_length'] / 2
        rotation = '' if c['lock_rotation'] else '<joint name="angle" type="hinge" axis="0 1 0" damping="0.0001"/>'
        return f'''<mujoco model="minimal_recap"><compiler angle="radian"/>
<option timestep="{c['timestep']}" gravity="{c['gravity'][0]} {c['gravity'][1]} {c['gravity'][2]}" integrator="implicitfast"/>
<default><geom friction="{c['friction']} 0.005 0.0001" solref="0.006 1" solimp="0.95 0.99 0.001"/>
<joint damping="0.1"/></default>
<worldbody><light pos="0 -1 1"/><camera name="front" pos="0 -1.1 0.55" xyaxes="1 0 0 0 0.4 1"/>
<geom name="table" type="plane" size="0.6 0.3 0.02" rgba="0.7 0.7 0.7 1"/>
<geom name="wall" type="box" pos="0 0 {s['wall_height']/2}" size="{c['wall_width']/2} 0.08 {s['wall_height']/2}" rgba="0.5 0.3 0.2 1"/>
<geom name="target" type="box" pos="{s['goal_x']} 0 0.0002" size="{c['target_half_width']} 0.05 0.0002" contype="0" conaffinity="0" rgba="0.2 0.8 0.3 0.5"/>
<body name="block" pos="{s['object_x']} 0 {b}"><joint name="ox" type="slide" axis="1 0 0"/><joint name="oz" type="slide" axis="0 0 1"/>{rotation}
<geom name="object" type="box" size="{b} {b} {b}" mass="{c['block_mass']}" rgba="0.9 0.4 0.1 1"/></body>
<body name="gripper"><joint name="gx" type="slide" axis="1 0 0" range="{c['workspace_x'][0]} {c['workspace_x'][1]}" limited="true"/>
<joint name="gz" type="slide" axis="0 0 1" range="{L+0.002} {c['max_height']}" limited="true"/>
<geom name="palm" type="box" pos="0 0 {L+0.008}" size="0.055 0.02 0.008" mass="0.2"/>
<body pos="{-f} 0 0"><joint name="left" type="slide" axis="-1 0 0" range="0 {c['max_opening']/2}" limited="true"/><geom name="lf" type="box" size="{f} 0.022 {L}" mass="0.04" rgba="0.2 0.3 0.8 1"/></body>
<body pos="{f} 0 0"><joint name="right" type="slide" axis="1 0 0" range="0 {c['max_opening']/2}" limited="true"/><geom name="rf" type="box" size="{f} 0.022 {L}" mass="0.04" rgba="0.2 0.3 0.8 1"/></body></body></worldbody>
<equality><joint joint1="left" joint2="right" polycoef="0 1 0 0 0" solref="0.004 1"/></equality>
<actuator><position joint="gx" kp="{c['servo_kp']}" kv="{c['servo_kv']}" forcerange="-{c['servo_force']} {c['servo_force']}"/>
<position joint="gz" kp="{c['servo_kp']}" kv="{c['servo_kv']}" forcerange="-{c['servo_force']} {c['servo_force']}"/>
<position joint="left" kp="{c['grip_kp']}" kv="3" forcerange="-{c['grip_force']} {c['grip_force']}"/>
<position joint="right" kp="{c['grip_kp']}" kv="3" forcerange="-{c['grip_force']} {c['grip_force']}"/></actuator></mujoco>'''

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.seed = seed
        c = self.c
        layout = self.np_random.choice(c['layout'])
        a, b = layout.split('_')
        self.scene = {'layout': layout, 'object_x': -float(self.np_random.uniform(*c[a])),
                      'goal_x': float(self.np_random.uniform(*c[b])),
                      'wall_height': float(self.np_random.uniform(*c['wall_height']))}
        if options and 'scene' in options:
            self.scene = copy.deepcopy(options['scene'])
        if self.renderer:
            self.renderer.close()
            self.renderer = None
        self.model = mujoco.MjModel.from_xml_string(self.xml())
        self.data = mujoco.MjData(self.model)
        self.j = {n: self.model.joint(n).qposadr[0] for n in ['gx', 'gz', 'left', 'right', 'ox', 'oz']}
        self.v = {n: self.model.joint(n).dofadr[0] for n in self.j}
        self.data.qpos[self.j['gx']] = self.scene['object_x']
        self.data.qpos[self.j['gz']] = 0.20
        self.data.qpos[self.j['left']] = self.data.qpos[self.j['right']] = c['max_opening']/2
        self.data.ctrl[:] = [self.scene['object_x'], 0.20, c['max_opening']/2, c['max_opening']/2]
        mujoco.mj_forward(self.model, self.data)
        self.steps = self.hold = self.grasps = self.drops = 0
        self.ever_grasped = self.was_grasped = self.first_grasp = False
        self.close_attempts = 0
        self.previous_close = False
        self.done = False
        return self.obs(), self.info('running')

    @property
    def grip(self):
        return np.array([self.data.qpos[self.j[n]] for n in ['gx', 'gz']])

    @property
    def block(self):
        return self.data.body('block').xpos[[0, 2]].copy()

    @property
    def opening(self):
        return sum(self.data.qpos[self.j[n]] for n in ['left', 'right'])

    def contact(self, a, b):
        a, b = self.model.geom(a).id, self.model.geom(b).id
        return any({int(x.geom1), int(x.geom2)} == {a, b} for x in self.data.contact)

    def grasped(self):
        return self.contact('object', 'lf') and self.contact('object', 'rf')

    def angle(self):
        return 0. if self.c['lock_rotation'] else self.data.qpos[self.model.joint('angle').qposadr[0]]

    def obs(self):
        qv = self.data.qvel
        goal = np.array([self.scene['goal_x'], self.c['block_size']/2])
        groups = {
            'gripper': [*self.grip, *[qv[self.v[n]] for n in ['gx', 'gz']], self.opening,
                        qv[self.v['left']] + qv[self.v['right']]],
            'object': [*self.block, qv[self.v['ox']], qv[self.v['oz']], np.sin(self.angle()), np.cos(self.angle()),
                       0. if self.c['lock_rotation'] else qv[self.model.joint('angle').dofadr[0]]],
            'goal': goal, 'wall': [0., self.c['wall_width'], self.scene['wall_height']],
            'relative': [*(self.grip-self.block), *(self.block-goal)]}
        return np.concatenate([np.asarray(groups[k]) for k in self.cfg['observation']['features']]).astype(np.float32)

    def step(self, action):
        if self.done:
            raise RuntimeError('reset required after task termination')
        action = np.asarray(action, dtype=float)
        if action.shape != (3,) or not np.isfinite(action).all():
            raise ValueError('Expected finite 3D action')
        self.executed_action = np.clip(action, -1, 1).astype(np.float32)
        target = self.grip + self.executed_action[:2]*self.c['max_delta']
        target[0] = np.clip(target[0], *self.c['workspace_x'])
        target[1] = np.clip(target[1], self.c['finger_length']/2+0.002, self.c['max_height'])
        opening = (self.executed_action[2]+1)*self.c['max_opening']/2
        self.data.ctrl[:] = [*target, opening/2, opening/2]
        closing = self.executed_action[2] < -0.5
        if closing and not self.previous_close:
            self.close_attempts += 1
        self.previous_close = closing
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError('Nonfinite physics')
        self.steps += 1
        grasp = self.grasped() and self.block[1] > self.c['block_size']*0.8
        if grasp and not self.was_grasped:
            self.grasps += 1
            if not self.ever_grasped:
                self.first_grasp = self.close_attempts == 1
            self.ever_grasped = True
        if self.was_grasped and not grasp:
            self.drops += 1
        self.was_grasped = grasp
        speed = np.linalg.norm(self.data.qvel[[self.v['ox'], self.v['oz']]])
        angular = 0. if self.c['lock_rotation'] else abs(self.data.qvel[self.model.joint('angle').dofadr[0]])
        success_now = (abs(self.block[0]-self.scene['goal_x']) + self.c['block_size']/2*(abs(np.sin(self.angle()))+abs(np.cos(self.angle()))) <= self.c['target_half_width']
                       and self.contact('object', 'table') and speed < self.c['success_speed']
                       and angular < self.c['success_angular_speed'] and self.opening > self.c['block_size']+0.008
                       and not self.contact('object', 'lf') and not self.contact('object', 'rf'))
        self.hold = self.hold+1 if success_now else 0
        success = self.hold >= self.c['success_hold_steps']
        failure = self.block[0] < self.c['workspace_x'][0] or self.block[0] > self.c['workspace_x'][1] or self.block[1] < -0.03
        timeout = self.budget is not None and self.steps >= self.budget
        reason = 'success' if success else 'out_of_bounds' if failure else 'task_timeout' if timeout else 'running'
        self.done = success or failure or timeout
        if self.done and not success and self.failure_cost is None:
            reward = np.nan  # expert failures are audit-only, never training targets
        else:
            reward = 0. if success else -float(self.failure_cost) if self.done else -1.
        return self.obs(), reward, self.done, False, self.info(reason)

    def info(self, reason):
        return {'reason': reason, 'success': reason == 'success', 'steps': self.steps, 'scene': copy.deepcopy(self.scene),
                'first_grasp_success': self.first_grasp, 'grasps': self.grasps, 'regrasps': max(0, self.grasps-1),
                'drops': self.drops, 'recovered': reason == 'success' and (self.close_attempts > 1 or self.drops > 1)}

    def snapshot(self):
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(self.model, spec))
        mujoco.mj_getState(self.model, self.data, state, spec)
        keys = ['steps', 'hold', 'grasps', 'drops', 'ever_grasped', 'was_grasped', 'first_grasp', 'close_attempts', 'previous_close', 'done']
        return {'physics': state.tolist(), 'scene': copy.deepcopy(self.scene), 'seed': self.seed,
                'management': {k: getattr(self, k) for k in keys}, 'rng': copy.deepcopy(self.np_random.bit_generator.state),
                'budget': self.budget}

    def restore(self, state):
        self.reset(seed=state['seed'], options={'scene': state['scene']})
        mujoco.mj_setState(self.model, self.data, np.array(state['physics']), mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(self.model, self.data)
        for k, v in state['management'].items():
            setattr(self, k, v)
        self.np_random.bit_generator.state = copy.deepcopy(state['rng'])
        return self.obs()

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.renderer.update_scene(self.data, camera='front')
        return self.renderer.render()

    def close(self):
        if self.renderer:
            self.renderer.close()
            self.renderer = None
