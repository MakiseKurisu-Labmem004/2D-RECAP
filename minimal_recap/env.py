import copy
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces


class ManipulationEnv(gym.Env):
    """Shared numerical observations, contacts and resumable MuJoCo state."""
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, config, budget=None, failure_cost=None):
        self.cfg = config
        self.c = config['environment']
        self.budget = budget
        self.failure_cost = failure_cost
        self.action_space = spaces.Box(-1., 1., (4,), dtype=np.float32)
        self.renderer = None
        dt = self.c['timestep']
        self.substeps = round(1 / self.c['control_hz'] / dt)
        if not np.isclose(self.substeps * dt * self.c['control_hz'], 1):
            raise ValueError('Control period must be an integer multiple of timestep')
        self.reset(seed=0)
        self.observation_space = spaces.Box(-np.inf, np.inf, self.obs().shape, dtype=np.float32)

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
        return self.data.qpos[self.model.joint('angle').qposadr[0]]

    def info(self, reason):
        return {'reason': reason, 'success': reason == 'success', 'steps': self.steps, 'scene': copy.deepcopy(self.scene),
                'first_grasp_success': self.first_grasp, 'grasps': self.grasps, 'regrasps': max(0, self.grasps-1),
                'drops': self.drops, 'recovered': reason == 'success' and (self.close_attempts > 1 or self.drops > 0)}

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
