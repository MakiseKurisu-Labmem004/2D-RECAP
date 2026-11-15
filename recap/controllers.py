"""Expert tracking of collision-checked joint paths."""

import numpy as np


def quintic_reference(start, goal, duration, time):
    u = np.clip(time / duration, 0., 1.)
    delta = np.asarray(goal) - start
    position = np.asarray(start) + delta * (10*u**3 - 15*u**4 + 6*u**5)
    velocity = delta * (30*u**2 - 60*u**3 + 30*u**4) / duration
    acceleration = delta * (60*u - 180*u**2 + 120*u**3) / duration**2
    return position, velocity, acceleration


class WaypointExpert:
    kind = 'waypoint'

    def __init__(self, path, cfg):
        self.path = np.asarray(path)
        self.index = 1
        self.limit = cfg['env']['action_limit']

    def __call__(self, obs, env):
        if np.max(abs(self.path[self.index] - env.data.qpos)) < .025 and self.index < len(self.path)-1:
            self.index += 1
        return np.clip(self.path[self.index] - env.data.qpos, -self.limit, self.limit)


class QuinticExpert:
    kind = 'quintic'

    def __init__(self, path, cfg):
        self.path = np.asarray(path, dtype=float)
        self.limit = cfg['env']['action_limit']
        self.dt = cfg['env']['timestep'] * cfg['env']['substeps']
        self.feedforward = cfg['env']['damping'] / cfg['env']['kp']
        planner = cfg['planner']
        speed = planner.get('expert_max_velocity', .6)
        acceleration = planner.get('expert_max_acceleration', 2.)
        self.tolerance = planner.get('expert_waypoint_tolerance', .003)
        if not all(np.isfinite(value) and value > 0 for value in (speed, acceleration, self.tolerance)):
            raise ValueError('Expert velocity, acceleration and waypoint tolerance must be positive and finite')
        if self.path.ndim != 2 or self.path.shape[1] != 2 or len(self.path) < 2 or not np.isfinite(self.path).all():
            raise ValueError('Expert path must contain at least two finite joint poses')
        spans = np.max(abs(np.diff(self.path, axis=0)), axis=1)
        durations = np.maximum(1.875 * spans / speed, np.sqrt((10 / np.sqrt(3)) * spans / acceleration))
        self.durations = np.maximum(1, np.ceil(durations / self.dt)).astype(int) * self.dt
        self.index = 0
        self.time = 0.

    def __call__(self, obs, env):
        if self.index >= len(self.durations):
            return np.clip(self.path[-1] - env.data.qpos, -self.limit, self.limit)
        duration = self.durations[self.index]
        self.time = min(duration, self.time + self.dt)
        target, velocity, _ = quintic_reference(self.path[self.index], self.path[self.index+1], duration, self.time)
        # Compensate joint damping while retaining the same position-actuator action.
        command = target + self.feedforward * velocity
        if self.time >= duration and np.max(abs(self.path[self.index+1] - env.data.qpos)) < self.tolerance:
            self.index += 1
            self.time = 0.
        return np.clip(command - env.data.qpos, -self.limit, self.limit)


def make_expert(path, cfg):
    kind = cfg['planner'].get('expert_controller', 'waypoint')
    controllers = dict(waypoint=WaypointExpert, quintic=QuinticExpert)
    if kind not in controllers:
        raise ValueError('expert_controller must be waypoint or quintic')
    return controllers[kind](path, cfg)
