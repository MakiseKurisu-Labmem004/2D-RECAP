import heapq
import numpy as np
from tqdm import tqdm
from .env import clearance, points, ObstacleReach2D
from .sampling import sample_scene
from .difficulty import DIFFICULTIES,difficulty_quotas
from .proposals import propose_obstruction
from .controllers import make_expert,expert_fits_budget
from .splines import UnsafeSplineError


def ik(goal, limit):
    c = (np.dot(goal, goal)-.5)/.5
    if abs(c)>1:
        return []
    result = []
    for q2 in [np.arccos(c), -np.arccos(c)]:
        q1 = np.arctan2(goal[1],goal[0])-np.arctan2(np.sin(q2),1+np.cos(q2))
        q1 = (q1+np.pi)%(2*np.pi)-np.pi
        q = np.array([q1,q2])
        if np.all(abs(q)<=limit): result.append(q)
    return result


def safe_edge(a,b,scene,cfg):
    n = max(2,int(np.ceil(np.linalg.norm(b-a)/cfg['planner']['interpolation']))+1)
    return all(clearance(q,scene,cfg['env']['link_radius']).min()>cfg['env']['margin'] for q in np.linspace(a,b,n))


def local_medium_detour(start, goal, scene, cfg):
    delta = goal - start
    distance = np.linalg.norm(delta)
    if distance < 1e-8:
        return None
    normal = np.array([-delta[1], delta[0]]) / distance
    middle = (start + goal) / 2
    for scale in (.15, .30, .45):
        for sign in (-1, 1):
            waypoint = middle + sign * scale * distance * normal
            if np.any(np.abs(waypoint) > cfg['env']['joint_limit']):
                continue
            path = np.array([start, waypoint, goal])
            ratio = np.linalg.norm(np.diff(path, axis=0), axis=1).sum() / distance
            if ratio >= cfg['planner']['hard_ratio']:
                continue
            if safe_edge(start, waypoint, scene, cfg) and safe_edge(waypoint, goal, scene, cfg):
                return path
    return None


def plan(start,goal,scene,cfg):
    if safe_edge(start,goal,scene,cfg): return np.array([start,goal])
    size = cfg['planner']['resolution']
    limit = cfg['env']['joint_limit']
    grid = np.linspace(-limit,limit,size)
    index = lambda q: tuple(np.clip(np.rint((q+limit)/(2*limit)*(size-1)).astype(int),0,size-1))
    coord = lambda k: np.array([grid[k[0]],grid[k[1]]])
    s,t = index(start),index(goal)
    if not safe_edge(start,coord(s),scene,cfg) or not safe_edge(coord(t),goal,scene,cfg): return None
    queue=[(0.,s)]; cost={s:0.}; parent={}; closed=set()
    while queue:
        _,u=heapq.heappop(queue)
        if u in closed: continue
        if u==t:
            path=[goal,coord(t)]
            while u!=s:
                u=parent[u]; path.append(coord(u))
            path.append(start); path=path[::-1]
            # Greedy shortcutting; every shortcut receives the same swept geometry check.
            smooth=[path[0]]; i=0
            while i<len(path)-1:
                j=len(path)-1
                while j>i+1 and not safe_edge(path[i],path[j],scene,cfg): j-=1
                smooth.append(path[j]); i=j
            return np.array(smooth)
        closed.add(u)
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
            v=(u[0]+dx,u[1]+dy)
            if not (0<=v[0]<size and 0<=v[1]<size) or v in closed: continue
            d=np.linalg.norm(coord(v)-coord(u)); nc=cost[u]+d
            if nc<cost.get(v,float('inf')) and safe_edge(coord(u),coord(v),scene,cfg):
                cost[v]=nc; parent[v]=u
                heapq.heappush(queue,(nc+np.linalg.norm(coord(v)-coord(t)),v))
    return None


def completed_goal_path(path, episode, scene, cfg):
    # A Cartesian goal has two IK solutions. Stop at the solution the rollout
    # actually reached, rather than retaining an unused, more distant endpoint.
    goals = ik(np.asarray(scene['goal']), cfg['env']['joint_limit'])
    if not goals: return path
    goal = min(goals, key=lambda pose: np.linalg.norm(pose-episode['qpos'][-1]))
    best = path
    length = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
    for index, vertex in enumerate(path):
        if not safe_edge(vertex, goal, scene, cfg): continue
        candidate = np.vstack((path[:index+1], goal))
        cost = np.linalg.norm(np.diff(candidate, axis=0), axis=1).sum()
        if cost < length - 1e-8: best, length = candidate, cost
    return best


def rollout(env,scene,policy,seed,version,noise=0.,limit=None,reset=True,until_done=False):
    import time
    rng=np.random.default_rng(seed)
    if reset: obs,_=env.reset(seed=seed,options={'scene':scene})
    else:
        if env.done: raise ValueError('Cannot continue a terminated environment')
        obs=env.obs()
    rows=[]
    remaining=env.cfg['horizon']-env.steps
    if until_done and limit is not None: raise ValueError('until_done does not accept a step limit')
    from itertools import count
    indices=count() if until_done else range(min(limit or remaining,remaining))
    for t in indices:
        action=np.clip(policy(obs,env)+rng.normal(0,noise,2),-env.cfg['action_limit'],env.cfg['action_limit'])
        nxt,r,term,trunc,info=env.step(action,allow_timeout=False) if until_done else env.step(action)
        rows.append(dict(observation=obs,action=action,next_observation=nxt,reward=r,terminated=term,
                         truncated=trunc,success=info['success'],failure_reason=info['failure_reason'],
                         applied_command=info['applied_command'],step_index=t,timestamp=time.time(),
                         qpos=env.data.qpos.copy(),clearance=info['clearance'],distance=info['distance'],
                         goal_hold_steps=env.held,control_commands=info['control_commands'],
                         substep_qpos=info['substep_qpos'],executed_substeps=info['executed_substeps']))
        obs=nxt
        if term or trunc: break
    ep={k:np.asarray([row[k] for row in rows]) for k in rows[0]}
    from .rewards import REWARD_VERSION
    ep['reward_version']=np.array(REWARD_VERSION)
    ep.update(scene_id=np.array(scene['scene_id']),episode_id=np.array(f'{version}-{seed}'),seed=np.array(seed),
              policy_version=np.array(version),action_source=np.array('expert' if version=='expert' else 'policy'),
              exploration=np.array(noise),success_distance=np.array(env.cfg['success_distance']),
              strict_success_distance=np.array(min(env.cfg['success_distance'],env.cfg.get('strict_success_distance',.03))),
              goal_hold_required=np.array(env.cfg['hold']),
              control_dt=np.array(env.cfg['timestep']*env.cfg['substeps']),physics_timestep=np.array(env.cfg['timestep']))
    return ep


class ScenarioGenerator:
    def __init__(self,cfg):
        self.cfg=cfg; self.rejections={}; self.difficulty_summary={}
    def generate(self,split,count,seed,initial_pairs=None):
        ratios=self.cfg.get('difficulty_ratio')
        quotas=difficulty_quotas(count,ratios) if ratios is not None else None
        method=self.cfg['planner'].get('proposal','random')
        if method not in ('random','guided'):
            raise ValueError('planner.proposal must be random or guided')
        counts=dict.fromkeys(DIFFICULTIES,0)
        accepted=list(initial_pairs or [])
        for scene,episode in accepted:
            counts[scene['metrics']['difficulty']]+=1
        if len(accepted)>count or (quotas is not None and any(counts[name]>quotas[name] for name in DIFFICULTIES)):
            raise ValueError(f'{split}: existing scenes exceed the requested difficulty quotas')
        existing_ids={episode['episode_id'].item() for scene,episode in accepted}
        summary=dict(target=quotas,accepted=counts,attempts=0)
        self.difficulty_summary[split]=summary
        previous_rejections=self.rejections.copy()
        guided=method=='guided' and quotas is not None
        if guided:
            summary['proposed']=dict.fromkeys(DIFFICULTIES,0)
            summary['local_detours']=0
        rng=np.random.default_rng(seed); env=ObstacleReach2D(self.cfg['env'])
        progress=tqdm(total=count,initial=len(accepted),desc=f'Generate {split}',unit='scene',dynamic_ncols=True)
        postfix={name:f'{counts[name]}/{quotas[name]}' for name in DIFFICULTIES} if quotas is not None else {}
        try:
            for attempt in range(self.cfg['planner']['max_attempts']):
                if len(accepted)==count: return accepted
                summary['attempts']=attempt+1
                if f'expert-{seed+attempt}' in existing_ids: continue
                postfix['attempt']=attempt+1
                progress.set_postfix(postfix)
                scene=sample_scene(rng,split,self.cfg,seed,attempt)
                goals=ik(np.array(scene['goal']),self.cfg['env']['joint_limit'])
                if guided:
                    remaining=np.array([quotas[name]-counts[name] for name in DIFFICULTIES],dtype=float)
                    requested=rng.choice(DIFFICULTIES,p=remaining/remaining.sum())
                    summary['proposed'][requested]+=1
                    if goals:
                        goal=goals[rng.integers(len(goals))]
                        scene=propose_obstruction(scene,goal,requested,rng,self.cfg)
                        goals=[goal]
                    if scene is None or not goals:
                        self.rejections['construction_failed']=self.rejections.get('construction_failed',0)+1
                        continue
                q0=np.array(scene['q0'])
                reason='initial_collision'
                if clearance(q0,scene,self.cfg['env']['link_radius']).min()>self.cfg['env']['margin']:
                    reason='planning_or_execution'
                    for goal in goals:
                        path=None
                        if guided and requested=='medium':
                            path=local_medium_detour(q0,goal,scene,self.cfg)
                            if path is not None: summary['local_detours']+=1
                        if path is None: path=plan(q0,goal,scene,self.cfg)
                        if path is None: continue
                        length=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())
                        ratio=length/max(np.linalg.norm(goal-q0),1e-8)
                        direct=safe_edge(q0,goal,scene,self.cfg)
                        if guided and requested=='medium' and not direct and ratio>=self.cfg['planner']['hard_ratio']:
                            reason='medium_detour_too_long'
                            continue
                        if quotas is not None and (
                            (direct and counts['easy']>=quotas['easy']) or
                            (not direct and ratio<self.cfg['planner']['hard_ratio'] and counts['medium']>=quotas['medium'])
                        ):
                            reason='difficulty_quota_full'
                            continue
                        try: expert=make_expert(path,self.cfg,scene)
                        except UnsafeSplineError:
                            reason='spline_infeasible'
                            continue
                        if not expert_fits_budget(expert,self.cfg):
                            reason='spline_too_long'
                            continue
                        ep=rollout(env,scene,expert,seed+attempt,'expert',limit=self.cfg['planner']['expert_limit'])
                        if not ep['success'][-1]: continue
                        if expert.kind in ('quintic','spline'):
                            completed=completed_goal_path(path,ep,scene,self.cfg)
                            if completed.shape!=path.shape or not np.allclose(completed,path,atol=1e-8,rtol=0):
                                path=completed
                                try: expert=make_expert(path,self.cfg,scene)
                                except UnsafeSplineError:
                                    reason='spline_infeasible'
                                    continue
                                if not expert_fits_budget(expert,self.cfg):
                                    reason='spline_too_long'
                                    continue
                                ep=rollout(env,scene,expert,seed+attempt,'expert',limit=self.cfg['planner']['expert_limit'])
                                if not ep['success'][-1]: continue
                            length=expert.trajectory.length if expert.kind=='spline' else float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())
                            ratio=length/max(np.linalg.norm(path[-1]-q0),1e-8)
                            direct=safe_edge(q0,path[-1],scene,self.cfg)
                        ep.update(expert_path=path,expert_controller=np.array(expert.kind))
                        if expert.kind=='spline':
                            ep.update(expert_spline_blend=np.array(expert.trajectory.blend),
                                      expert_spline_duration=np.array(expert.trajectory.duration))
                        gap=float(ep['clearance'].min())
                        difficulty='easy' if direct else 'hard' if ratio>=self.cfg['planner']['hard_ratio'] and gap<self.cfg['planner']['hard_clearance'] else 'medium'
                        if quotas is not None and counts[difficulty]>=quotas[difficulty]:
                            reason='difficulty_quota_full'
                            continue
                        scene['metrics']=dict(steps=len(ep['reward']),joint_path_length=length,
                            ee_path_length=float(np.linalg.norm(np.diff(np.array([points(q)[-1] for q in ep['qpos']]),axis=0),axis=1).sum()),
                            detour_ratio=ratio,min_clearance=gap,difficulty=difficulty)
                        accepted.append((scene,ep)); reason='accepted'
                        counts[difficulty]+=1
                        if quotas is not None:
                            postfix[difficulty]=f'{counts[difficulty]}/{quotas[difficulty]}'
                            progress.set_postfix(postfix,refresh=False)
                        progress.update(1)
                        break
                self.rejections[reason]=self.rejections.get(reason,0)+1
            if len(accepted)==count: return accepted
            missing={name:quotas[name]-counts[name] for name in DIFFICULTIES} if quotas is not None else {}
            split_rejections={name:value-previous_rejections.get(name,0) for name,value in self.rejections.items()
                              if value>previous_rejections.get(name,0)}
            raise RuntimeError(f'{split}: accepted {len(accepted)}/{count}; missing difficulty quotas: {missing}; rejection counts: {split_rejections}')
        finally:
            if guided:
                summary['rejections']={name:value-previous_rejections.get(name,0) for name,value in self.rejections.items()
                                       if value>previous_rejections.get(name,0)}
            progress.close()
            env.close()
