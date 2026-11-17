"""Task construction shared by training, evaluation, and saved-state replay."""
from .arm3r import Arm3REnv
from .upright import UprightExpert


def make_env(config, budget=None, failure_cost=None):
    task = config['environment']['task']
    if task == 'pedestal_upright' and config['environment']['robot'] == 'arm3r':
        return Arm3REnv(config, budget, failure_cost)
    raise ValueError('Only the arm3r pedestal_upright task is supported')


def make_expert(env):
    return UprightExpert(env)
