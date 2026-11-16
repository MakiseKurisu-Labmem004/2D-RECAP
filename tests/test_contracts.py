import copy
import numpy as np
import pytest
import torch
from minimal_recap.config import load
from minimal_recap.env import PlanarGraspEnv
from minimal_recap.expert import Expert
from minimal_recap.learning import Network, bin_targets, advantages
from minimal_recap.data import returns, freeze_contract, collect_episode, read_episode


@pytest.fixture
def c():
    return load('configs/experiments/smoke.yaml')


def episode(rewards, terminal=True):
    n = len(rewards)
    return {'reward': np.array(rewards), 'terminated': np.array([False]*(n-1)+[terminal]),
            'truncated': np.zeros(n, dtype=bool), 'metadata': {'status': 'complete', 'success': True}}


def test_return_and_short_terminal_advantage():
    ep = episode([-1, -1, 0])
    np.testing.assert_equal(returns(ep), [-2, -1, 0])
    np.testing.assert_allclose(advantages(ep, np.array([-3, -2, -1]), np.array([-2, -1, 999]), 10), [1, 1, 1])
    np.testing.assert_allclose(advantages(ep, np.array([-3, -2, -1]), np.array([-2, -1, 999]), 1), [0, 0, 1])
    ep['metadata']['status'] = 'aborted'
    with pytest.raises(ValueError):
        returns(ep)


def test_distribution_interpolation_and_overflow():
    support = torch.linspace(-400, 0, 201)
    y = torch.tensor([-400., -3., 0.])
    labels = bin_targets(y, support)
    torch.testing.assert_close(labels.sum(-1), torch.ones(3))
    torch.testing.assert_close((labels*support).sum(-1), y)
    with pytest.raises(ValueError):
        bin_targets(torch.tensor([-401.]), support)


def test_contract_ordering(c, tmp_path):
    contract = freeze_contract(c, [episode([-1]*240+[0])], tmp_path/'contract.json')
    assert contract['M'] == 241
    assert -contract['failure_cost'] < -(contract['M']-1)
    assert contract['support'][0] == -440
    c['reward']['failure_cost'] = 200
    with pytest.raises(ValueError):
        freeze_contract(c, [episode([-1]*240+[0])], tmp_path/'other.json')


def test_model_has_six_hidden_layers(c):
    net = Network(22, c['model'])
    assert sum(isinstance(m, torch.nn.Linear) for m in net.net) == 7


def test_snapshot_replay_and_action_interface(c):
    e = PlanarGraspEnv(c, 200, 200)
    obs, _ = e.reset(seed=32)
    assert e.observation_space.contains(obs)
    state = e.snapshot()
    a = np.array([0.5, -0.7, -1])
    first = e.step(a)
    e.restore(state)
    second = e.step(a)
    np.testing.assert_allclose(first[0], second[0], atol=1e-6)
    with pytest.raises(ValueError):
        e.step([np.nan, 0, 0])
    e.close()


def test_timeout_and_success_priority(c):
    e = PlanarGraspEnv(c, 1, 200)
    _, reward, terminated, truncated, info = e.step([0, 0, 1])
    assert terminated and not truncated and reward == -200 and info['reason'] == 'task_timeout'
    e.close()


@pytest.mark.parametrize('locked', [False, True])
def test_physical_expert(c, locked):
    c['environment']['lock_rotation'] = locked
    env = PlanarGraspEnv(c)
    obs, _ = env.reset(seed=10000)
    expert = Expert(env)
    lifted = False
    for _ in range(600):
        obs, _, done, _, info = env.step(expert(obs))
        lifted |= env.grasped() and env.block[1] > 0.10
        if done:
            break
    assert done and info['success'] and lifted
    env.close()


def test_aborted_is_resumable(c, tmp_path):
    env = PlanarGraspEnv(c, 10, 200)
    class Cancel:
        def __call__(self, obs):
            raise KeyboardInterrupt()
    path = tmp_path/'ep.npz'
    with pytest.raises(KeyboardInterrupt):
        collect_episode(env, Cancel(), 1, {'episode_id': 'x'}, path)
    ep = read_episode(path)
    assert ep['metadata']['status'] == 'aborted'
    assert not ep['metadata']['success']
    complete = collect_episode(env, lambda obs: np.array([0, 0, 1]), 1, {}, resume=ep)
    assert complete['metadata']['status'] == 'complete'
    assert len(complete['reward']) == 10
    env.close()

def test_success_has_priority_over_budget(c):
    c = copy.deepcopy(c)
    c['environment']['success_hold_steps'] = 1
    e = PlanarGraspEnv(c)
    scene = dict(e.scene, goal_x=e.scene['object_x'])
    e.reset(seed=10, options={'scene': scene})
    # Let the physical cube settle while gripper remains open above it.
    for _ in range(5):
        if e.done:
            break
        e.step([0, 0, 1])
    # Recreate hold requirement at a stable table-supported physical state.
    e.done = False
    e.hold = 0
    e.budget = e.steps+1
    e.failure_cost = 200
    _, r, terminal, trunc, info = e.step([0, 0, 1])
    assert terminal and not trunc and r == 0 and info['reason'] == 'success'
    e.close()


def test_training_paths_and_resume(c, tmp_path):
    from minimal_recap.learning import train, load_model
    from minimal_recap.config import merge
    from minimal_recap.data import episodes
    eps = []
    for seed in [10000, 10001]:
        env = PlanarGraspEnv(c, 200, 200)
        ep = collect_episode(env, Expert(env), seed, {'episode_id': f's{seed}', 'source': 'bc'}, tmp_path/f's{seed}.npz')
        assert ep['metadata']['success']
        eps.append(read_episode(tmp_path/f's{seed}.npz'))
        env.close()
    for seed in [10002, 10003]:
        env = PlanarGraspEnv(c, 200, 200)
        collect_episode(env, lambda obs: [0, 0, 1], seed, {'episode_id': f'f{seed}', 'source': 'bc'}, tmp_path/f'f{seed}.npz')
        eps.append(read_episode(tmp_path/f'f{seed}.npz'))
        env.close()
    c = merge(c, {k: {'epochs': 1, 'updates_per_epoch': 1} for k in ['bc_training', 'value_training', 'recap']})
    contract = {'support': [-399, 0], 'failure_cost': 200}
    initial = train(c, eps[:2], tmp_path/'bc.pt', 'bc', contract)
    val = train(c, eps, tmp_path/'v.pt', 'value', contract)
    labels = {e['metadata']['episode_id']: np.ones(len(e['reward'])) for e in eps}
    for method in ['conditioned', 'aggregate_bc', 'success_bc', 'unconditioned']:
        config = merge(c, {'recap': {'method': method}})
        model = train(config, eps, tmp_path/f'{method}.pt', 'update', contract, initial=initial, experts=eps[:2], labels=labels)
        assert model.conditional == (method == 'conditioned')
        again = train(config, eps, tmp_path/f'{method}.pt', 'update', contract, initial=initial, experts=eps[:2], labels=labels)
        for a, b in zip(model.parameters(), again.parameters()):
            torch.testing.assert_close(a, b)
