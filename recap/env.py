import numpy as np
import gymnasium as gym
import mujoco

from .geometry import clearance, points
from .observations import OBSERVATION_DIM
from .control_filter import C3CommandFilter


class ObstacleReach2D(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 20}

    def __init__(self, config, render_mode=None):
        self.cfg = c = config
        for key in ('horizon', 'substeps', 'hold'):
            if not isinstance(c[key], int) or isinstance(c[key], bool) or c[key] <= 0:
                raise ValueError(f'{key} must be a positive integer')
        for key in ('timestep', 'action_limit', 'success_distance'):
            if not np.isfinite(c[key]) or c[key] <= 0:
                raise ValueError(f'{key} must be positive and finite')
        self.metadata = {**type(self).metadata, 'render_fps': round(1/(c['timestep']*c['substeps']))}
        self.render_mode = render_mode
        self.viewer = self.renderer = None
        self.interpolation = c.get('control_interpolation', 'hold')
        if self.interpolation not in ('hold', 'c3_filter'):
            raise ValueError('control_interpolation must be hold or c3_filter')
        self.action_space = gym.spaces.Box(-c['action_limit'], c['action_limit'], (2,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBSERVATION_DIM,), np.float32)
        self.model = mujoco.MjModel.from_xml_string(f'''<mujoco>
        <option timestep="{c['timestep']}" gravity="0 0 0"/>
        <default><joint type="hinge" axis="0 0 1" damping="{c['damping']}" range="{-c['joint_limit']} {c['joint_limit']}"/>
        <geom type="capsule" size="{c['link_radius']}" density="1000"/></default>
        <compiler angle="radian"/>
        <worldbody><light pos="0 0 3"/><camera name="top" pos="0 0 2.7"/>
        <geom name="obstacle" type="cylinder" pos=".5 .2 0" size=".07 .05" rgba=".9 .2 .2 1"/>
        <site name="goal" pos=".7 .3 0" size="{c['success_distance']}" rgba=".2 .9 .2 1"/>
        <body><joint name="q1"/><geom name="link1" fromto="0 0 0 .5 0 0"/>
        <body pos=".5 0 0"><joint name="q2"/><geom name="link2" fromto="0 0 0 .5 0 0"/></body></body>
        </worldbody><actuator><position joint="q1" kp="{c['kp']}" forcerange="{-c['torque']} {c['torque']}"/>
        <position joint="q2" kp="{c['kp']}" forcerange="{-c['torque']} {c['torque']}"/></actuator></mujoco>''')
        self.data = mujoco.MjData(self.model)

    def obs(self):
        return np.concatenate([np.sin(self.data.qpos), np.cos(self.data.qpos),
                               self.scene['goal'], self.scene['obstacle'], [self.scene['radius']]]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.scene = options['scene'] if options else dict(q0=[0., 1.], goal=[.6,.6], obstacle=[-.4,.3], radius=.07)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.scene['q0']
        self.data.ctrl[:] = self.data.qpos
        self.command_filter = C3CommandFilter(self.data.qpos, self.cfg.get('control_filter_tau', .003)) if self.interpolation == 'c3_filter' else None
        self.model.geom_pos[0, :2] = self.scene['obstacle']
        self.model.geom_size[0, 0] = self.scene['radius']
        self.model.site_pos[0, :2] = self.scene['goal']
        mujoco.mj_forward(self.model, self.data)
        self.steps = self.held = 0
        self.done = False
        return self.obs(), {}

    def step(self, action, *, allow_timeout=True):
        if self.done:
            raise RuntimeError('reset required after episode end')
        action = np.asarray(action)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError('action must be finite with shape (2,)')
        c = self.cfg
        command = np.clip(self.data.qpos + np.clip(action, -c['action_limit'], c['action_limit']), -c['joint_limit'], c['joint_limit'])
        reason = ''
        min_gap = float('inf')
        commands = np.empty((c['substeps'], 2))
        poses = np.empty((c['substeps'], 2))
        executed = 0
        for index in range(c['substeps']):
            self.data.ctrl[:] = self.command_filter.advance(command, c['timestep']) if self.command_filter is not None else command
            commands[index] = self.data.ctrl
            executed = index+1
            mujoco.mj_step(self.model, self.data)
            poses[index] = self.data.qpos
            gaps = clearance(self.data.qpos, self.scene, c['link_radius'])
            min_gap = min(min_gap, float(gaps.min()))
            if gaps.min() <= 0:
                reason = f'link{gaps.argmin()+1}_collision'
                break
            if not np.isfinite(self.data.qpos).all() or np.any(abs(self.data.qpos)>c['joint_limit']+.02):
                reason = 'constraint'
                break
        commands[executed:] = self.data.ctrl
        poses[executed:] = self.data.qpos
        self.steps += 1
        distance = float(np.linalg.norm(points(self.data.qpos)[-1]-self.scene['goal']))
        self.held = self.held+1 if distance < c['success_distance'] else 0
        success = self.held >= c['hold'] and not reason
        terminated = bool(success or reason)
        truncated = allow_timeout and self.steps >= c['horizon'] and not terminated
        if truncated:
            reason = 'timeout'
        self.done = terminated or truncated
        return self.obs(), (0. if success else -150. if self.done else -1.), terminated, truncated, dict(
            success=bool(success), failure_reason=reason, distance=distance, collision='collision' in reason,
            applied_command=command.copy(), control_commands=commands, substep_qpos=poses,
            executed_substeps=executed, clearance=min_gap)

    def render(self):
        if self.render_mode == 'human':
            if self.viewer is None:
                from mujoco import viewer
                self.viewer = viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
        else:
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.renderer.update_scene(self.data, camera='top')
            return self.renderer.render()

    def close(self):
        for obj in (self.viewer, self.renderer):
            if obj is not None:
                obj.close()

