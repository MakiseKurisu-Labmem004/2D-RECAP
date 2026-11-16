import numpy as np


class Expert:
    """Closed-loop controller; all effects go through env.step(normalized_action)."""
    def __init__(self, env):
        self.env = env
        self.stage = 'open'
        self.ticks = 0
        self.probe_start = None
        self.attempts = 0
        self.safe = 0.

    def state_dict(self):
        return {k: getattr(self, k) for k in ['stage', 'ticks', 'probe_start', 'attempts', 'safe']}

    def load_state_dict(self, state):
        for k, v in state.items():
            setattr(self, k, v)

    def switch(self, stage):
        self.stage, self.ticks = stage, 0

    def __call__(self, obs):
        e, c = self.env, self.env.c
        p, b = e.grip, e.block
        self.ticks += 1
        # Rotation envelope plus both finger and object lower extents; intentionally conservative.
        self.safe = max(e.scene['wall_height'] + c['finger_length']/2 + c['block_size']/np.sqrt(2)
                        + e.cfg['expert_rollout']['clearance'], 0.16)
        if self.safe > c['max_height']-0.01:
            raise ValueError('Scene lacks vertical transport clearance')
        target, opening = p.copy(), c['max_opening']
        grasp_z = max(c['finger_length']/2+0.006, b[1]+0.016)
        stage = self.stage
        if stage == 'open':
            target[1] = max(p[1], self.safe)
            if e.opening > c['max_opening']*0.9:
                self.switch('above')
        elif stage == 'above':
            target = np.array([b[0], self.safe])
            if np.max(np.abs(p-target)) < 0.005:
                self.switch('descend')
        elif stage == 'descend':
            target = np.array([b[0], grasp_z])
            if np.max(np.abs(p-target)) < 0.004:
                self.attempts += 1
                self.switch('close')
        elif stage == 'close':
            target, opening = np.array([b[0], grasp_z]), 0.
            if self.ticks >= e.cfg['expert_rollout']['close_steps']:
                self.probe_start = float(b[1])
                self.probe_target = float(p[1]+e.cfg['expert_rollout']['lift_probe'])
                self.switch('probe')
        elif stage == 'probe':
            # Derive fixed probe height from captured object height, avoiding moving target feedback.
            target, opening = np.array([p[0], self.probe_start+0.016+e.cfg['expert_rollout']['lift_probe']]), 0.
            if self.ticks >= 12:
                self.switch('lift' if e.grasped() and b[1] > self.probe_start+0.015 else 'open')
        elif stage == 'lift':
            target, opening = np.array([p[0], self.safe]), 0.
            if not e.grasped():
                self.switch('open')
            elif abs(p[1]-self.safe) < 0.005:
                self.switch('transport')
        elif stage == 'transport':
            target, opening = np.array([e.scene['goal_x'], self.safe]), 0.
            if not e.grasped():
                self.switch('open')
            elif abs(p[0]-target[0]) < 0.004:
                self.switch('lower')
        elif stage == 'lower':
            target, opening = np.array([e.scene['goal_x'], c['finger_length']/2+0.006]), 0.
            if abs(p[1]-target[1]) < 0.004:
                self.switch('release')
        elif stage == 'release':
            target = np.array([e.scene['goal_x'], c['finger_length']/2+0.006])
            if e.opening > c['max_opening']*0.9:
                self.switch('wait')
        elif stage == 'wait':
            target = np.array([e.scene['goal_x'], self.safe])
            if self.ticks > 30 and abs(b[0]-e.scene['goal_x']) > c['target_half_width']-c['block_size']/2:
                self.switch('open')
        return np.array([*np.clip((target-p)/c['max_delta'], -1, 1), 2*opening/c['max_opening']-1], dtype=np.float32)
