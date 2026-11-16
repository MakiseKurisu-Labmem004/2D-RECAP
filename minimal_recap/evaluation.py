import copy
import numpy as np
import torch
from .env import PlanarGraspEnv
from .expert import Expert
from .learning import Behavior, advantages
from .data import collect_episode, returns, atomic_json


def suite_config(c, suite):
    out = copy.deepcopy(c)
    if suite == 'combination':
        out['environment']['layout'] = ['far_far']
    elif suite == 'height_1':
        out['environment']['wall_height'] = [0.11, 0.12]
    elif suite == 'height_2':
        out['environment']['wall_height'] = [0.13, 0.14]
    elif suite != 'id':
        raise ValueError(suite)
    return out


def summarize(eps, hz):
    n = len(eps)
    meta = [e['metadata'] for e in eps]
    success = np.array([m['success'] for m in meta], dtype=float)
    p = float(success.mean()) if n else 0.
    # Wilson interval, independent scene seeds, no claim of training-seed uncertainty.
    z, den = 1.96, 1+1.96**2/max(n, 1)
    center = (p+z*z/(2*max(n, 1)))/den
    half = z*np.sqrt(p*(1-p)/max(n, 1)+z*z/(4*max(n, 1)**2))/den
    return {'episodes': n, 'success_rate': p, 'success_ci95': [center-half, center+half],
            'first_grasp_success_rate': float(np.mean([m.get('first_grasp_success', False) for m in meta])) if n else 0.,
            'mean_regrasps': float(np.mean([m.get('regrasps', 0) for m in meta])) if n else 0.,
            'success_completion_seconds': float(np.mean([len(e['reward'])/hz for e in eps if e['metadata']['success']])) if success.sum() else None,
            'mean_return': float(np.mean([returns(e)[0] for e in eps])) if n else None}


def evaluate(c, model, contract, suite='id', validation=False, save_dir=None):
    config = suite_config(c, suite)
    seed = c['seeds']['validation'] if validation else c['seeds']['evaluation']
    count = c['bc_training']['validation_episodes'] if validation else c['evaluation']['episodes']
    eps = []
    env = PlanarGraspEnv(config, c['evaluation']['max_control_steps'], contract['failure_cost'])
    try:
        for i in range(count):
            ep = collect_episode(env, Behavior(model, c['evaluation']['exploration'], seed+i), seed+i,
                                 {'episode_id': f'{suite}_{i}', 'source': 'evaluation', 'suite': suite},
                                 save_dir / f'{suite}_{i:05}.npz' if save_dir else None)
            eps.append(ep)
    finally:
        env.close()
    return summarize(eps, c['environment']['control_hz'])


def physics_check(c, output, count=12):
    results = {}
    # Include all OOD geometries before reporting feasibility.
    for suite in ['id', 'combination', 'height_1', 'height_2']:
        cfg = suite_config(c, suite)
        cfg['environment']['layout'] = ['near_near', 'near_far', 'far_near'] if suite == 'id' else cfg['environment']['layout']
        rows = []
        env = PlanarGraspEnv(cfg, None)
        for i in range(count):
            obs, _ = env.reset(seed=c['seeds']['expert']+i)
            expert = Expert(env)
            # Validation guard is an explicit exception, never an expert task deadline.
            for t in range(1200):
                obs, _, term, _, info = env.step(expert(obs))
                if term:
                    break
            else:
                raise RuntimeError(f'Physics validation controller stalled: {suite}, seed={c["seeds"]["expert"]+i}, stage={expert.stage}')
            rows.append(info)
        results[suite] = {'successes': sum(x['success'] for x in rows), 'episodes': count, 'episodes_detail': rows}
        env.close()
    atomic_json(output, results)
    if any(v['successes'] != count for v in results.values()):
        raise RuntimeError('Physical feasibility gate failed; inspect physics.json before collecting data')
    return results


def make_recovery_pool(c, contract, output):
    states = []
    for mode in ['empty', 'drop']:
        env = PlanarGraspEnv(c, c['evaluation']['max_control_steps'], contract['failure_cost'])
        for i in range(c['evaluation']['pool_size']):
            seed = c['seeds']['diagnostics']+i
            obs, _ = env.reset(seed=seed)
            expert = Expert(env)
            injected = 0
            for t in range(c['evaluation']['max_control_steps']):
                action = expert(obs)
                if mode == 'empty' and expert.stage in ['descend', 'close', 'probe']:
                    action[0] = np.clip((env.scene['object_x']+0.06-env.grip[0])/env.c['max_delta'], -1, 1)
                    injected += 1
                if mode == 'drop' and (expert.stage == 'transport' or injected > 0):
                    action = np.array([0., 0., 1.])
                    injected += 1
                obs, _, done, _, _ = env.step(action)
                ready = (mode == 'empty' and injected >= 20 and not env.grasped()) or (mode == 'drop' and injected >= 8 and not env.grasped())
                if ready and not done:
                    states.append({'mode': mode, 'state': env.snapshot(), 'controller': {'expert': expert.state_dict(), 'perturbation': {'mode': mode, 'injected_steps': injected}},
                                   'remaining_budget': env.budget-env.steps})
                    break
                if done:
                    break
        env.close()
    if {x['mode'] for x in states} != {'empty', 'drop'}:
        raise RuntimeError('Could not generate both physical recovery modes')
    atomic_json(output, states)
    return states


def evaluate_pool(c, model, contract, pool):
    results = {}
    for mode in ['empty', 'drop']:
        eps = []
        for i, item in enumerate(pool):
            if item['mode'] != mode:
                continue
            env = PlanarGraspEnv(c, c['evaluation']['max_control_steps'], contract['failure_cost'])
            policy = Behavior(model, c['evaluation']['exploration'], c['seeds']['evaluation']+i)
            resume = {k: [] for k in ['observation', 'action', 'reward', 'next_observation', 'terminated', 'truncated']}
            resume.update(states=[item['state']], controller_states=[], metadata={'episode_id': f'pool_{i}', 'source': 'recovery', 'status': 'active'})
            ep = collect_episode(env, policy, item['state']['seed'], {}, resume=resume)
            eps.append(ep)
            env.close()
        results[mode] = summarize(eps, c['environment']['control_hz'])
        results[mode]['recovery_success_rate'] = results[mode]['success_rate']
    return results


def pregrasp_state(c, contract):
    env = PlanarGraspEnv(c, c['evaluation']['max_control_steps'], contract['failure_cost'])
    obs, _ = env.reset(seed=c['seeds']['diagnostics'])
    expert = Expert(env)
    for _ in range(env.budget):
        action = expert(obs)
        if expert.stage == 'close':
            return env, env.snapshot(), expert.state_dict()
        obs, _, done, _, _ = env.step(action)
        if done:
            break
    raise RuntimeError('No physical pregrasp state')


def diagnostics(c, policy, value, contract, output):
    import os
    os.environ.setdefault('MPLCONFIGDIR', str((output/'mpl_cache').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    env, base, controller = pregrasp_state(c, contract)
    exploration = c['evaluation']['exploration']
    stochastic = exploration['mode'] != 'none' and np.any(np.asarray(exploration['std']) > 0)
    repeats = c['evaluation']['diagnostic_repeats'] if stochastic else 1
    if stochastic and repeats < 2:
        raise ValueError('Stochastic reference requires multiple continuation samples')
    rows, rank_rows = [], []
    def v(obs):
        with torch.no_grad():
            return float(value.value(torch.tensor(obs, dtype=torch.float32)))
    def continue_return(state, prefix=None, repeat=0):
        env.restore(state)
        start = env.steps
        total, predicted = 0., v(env.obs())
        n = c['evaluation']['diagnostic_prefix_steps']
        if prefix is not None:
            for t in range(n):
                obs, r, done, _, _ = env.step(prefix(t))
                total += r
                if done:
                    break
            estimated = total+(0 if env.done else v(env.obs()))-predicted
        else:
            estimated = None
        behavior = Behavior(policy, exploration, c['seeds']['diagnostics']+repeat)
        while not env.done:
            _, r, _, _, _ = env.step(behavior(env.obs()))
            total += r
        return total, estimated, env.steps-start
    for offset in c['evaluation']['diagnostic_offsets']:
        env.restore(base)
        target_x = env.grip[0]+offset
        # Reach offsets via physical action, so every tested state is dynamically valid.
        for _ in range(8):
            env.step([np.clip((target_x-env.grip[0])/env.c['max_delta'], -1, 1), 0, 1])
        state = env.snapshot()
        if env.done:
            raise RuntimeError('Diagnostic setup exhausted budget')
        observed = [continue_return(state, repeat=j)[0] for j in range(repeats)]
        env.restore(state)
        rows.append({'offset': offset, 'actual_offset': float(env.grip[0]-env.block[0]), 'state': state, 'value': v(env.obs()), 'return_mean': float(np.mean(observed)),
                     'return_std': float(np.std(observed)), 'independent_continuations': repeats})
        for direction, name in [(0, 'direct'), (-np.sign(offset), 'correct'), (np.sign(offset), 'wrong')]:
            def prefix(t):
                return [direction if t < 4 else 0, 0, 1 if t < 4 else -1]
            samples = [continue_return(state, prefix, j) for j in range(repeats)]
            rank_rows.append({'offset': offset, 'fragment': name, 'return_mean': float(np.mean([x[0] for x in samples])),
                              'estimated_advantage': samples[0][1], 'prefix_steps': c['evaluation']['diagnostic_prefix_steps']})
    # Held-out complete policy episode for Value/Return/Advantage traces.
    ep = collect_episode(env, Behavior(policy, exploration, c['seeds']['diagnostics']), c['seeds']['diagnostics']+1,
                         {'episode_id': 'diagnostic_trace', 'source': 'diagnostic'}, output/'trace.npz')
    with torch.no_grad():
        vs = value.value(torch.tensor(np.asarray(ep['observation']), dtype=torch.float32)).numpy()
        ns = value.value(torch.tensor(np.asarray(ep['next_observation']), dtype=torch.float32)).numpy()
    ret = returns(ep)
    adv = advantages(ep, vs, ns, c['recap']['n_step'])
    fig, axes = plt.subplots(2, 1, figsize=(9, 6))
    axes[0].plot(vs, label='Value'); axes[0].plot(ret, label='Return'); axes[0].legend()
    axes[1].plot(adv, label='n-step Advantage'); axes[1].legend(); axes[1].set_xlabel('control step')
    fig.tight_layout(); fig.savefig(output/'trajectory.png'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r['offset'] for r in rows], [r['value'] for r in rows], 'o-', label='Value')
    ax.errorbar([r['offset'] for r in rows], [r['return_mean'] for r in rows], yerr=[r['return_std'] for r in rows], label='reference continuation')
    ax.legend(); ax.set_xlabel('physical horizontal offset (m)'); fig.tight_layout(); fig.savefig(output/'offsets.png'); plt.close(fig)
    pairs = []
    for offset in c['evaluation']['diagnostic_offsets']:
        group = [r for r in rank_rows if r['offset'] == offset]
        for i in range(len(group)):
            for j in range(i):
                dr = group[i]['return_mean']-group[j]['return_mean']
                da = group[i]['estimated_advantage']-group[j]['estimated_advantage']
                if abs(dr) > 1e-8:
                    pairs.append(float(np.sign(dr) == np.sign(da)))
    result = {'offsets': rows, 'fragments': rank_rows, 'trace_value_mae': float(np.mean(abs(vs-ret))),
              'advantage_pairwise_accuracy': float(np.mean(pairs)) if pairs else None,
              'reference_exploration': exploration, 'controller_setup': controller,
              'warning': 'Deadline is unobserved; V estimates the behavior-data conditional return, not alignment quality.'}
    atomic_json(output/'diagnostics.json', result)
    env.close()
    return result
