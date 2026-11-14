import heapq
import numpy as np
from tqdm import tqdm
from .env import clearance, points, ObstacleReach2D
from .sampling import sample_scene
from .difficulty import DIFFICULTIES,difficulty_quotas


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


def rollout(env,scene,policy,seed,version,noise=0.,limit=None):
    import time
    rng=np.random.default_rng(seed)
    obs,_=env.reset(seed=seed,options={'scene':scene})
    rows=[]
    for t in range(limit or env.cfg['horizon']):
        action=np.clip(policy(obs,env)+rng.normal(0,noise,2),-env.cfg['action_limit'],env.cfg['action_limit'])
        nxt,r,term,trunc,info=env.step(action)
        rows.append(dict(observation=obs,action=action,next_observation=nxt,reward=r,terminated=term,
                         truncated=trunc,success=info['success'],failure_reason=info['failure_reason'],
                         applied_command=info['applied_command'],step_index=t,timestamp=time.time(),
                         qpos=env.data.qpos.copy(),clearance=info['clearance']))
        obs=nxt
        if term or trunc: break
    ep={k:np.asarray([row[k] for row in rows]) for k in rows[0]}
    ep.update(scene_id=np.array(scene['scene_id']),episode_id=np.array(f'{version}-{seed}'),seed=np.array(seed),
              policy_version=np.array(version),action_source=np.array('expert' if version=='expert' else 'policy'),
              exploration=np.array(noise))
    return ep


class ScenarioGenerator:
    def __init__(self,cfg):
        self.cfg=cfg; self.rejections={}; self.difficulty_summary={}
    def generate(self,split,count,seed):
        ratios=self.cfg.get('difficulty_ratio')
        quotas=difficulty_quotas(count,ratios) if ratios is not None else None
        counts=dict.fromkeys(DIFFICULTIES,0)
        summary=dict(target=quotas,accepted=counts,attempts=0)
        self.difficulty_summary[split]=summary
        rng=np.random.default_rng(seed); env=ObstacleReach2D(self.cfg['env']); accepted=[]
        progress=tqdm(total=count,desc=f'Generate {split}',unit='scene',dynamic_ncols=True)
        postfix={name:f'0/{quotas[name]}' for name in DIFFICULTIES} if quotas is not None else {}
        try:
            for attempt in range(self.cfg['planner']['max_attempts']):
                if len(accepted)==count: return accepted
                summary['attempts']=attempt+1
                postfix['attempt']=attempt+1
                progress.set_postfix(postfix)
                scene=sample_scene(rng,split,self.cfg,seed,attempt)
                q0=np.array(scene['q0'])
                reason='initial_collision'
                if clearance(q0,scene,self.cfg['env']['link_radius']).min()>self.cfg['env']['margin']:
                    reason='planning_or_execution'
                    for goal in ik(np.array(scene['goal']),self.cfg['env']['joint_limit']):
                        path=plan(q0,goal,scene,self.cfg)
                        if path is None: continue
                        state=[1]
                        def expert(obs,env):
                            if np.max(abs(path[state[0]]-env.data.qpos))<.025 and state[0]<len(path)-1: state[0]+=1
                            return np.clip(path[state[0]]-env.data.qpos,-env.cfg['action_limit'],env.cfg['action_limit'])
                        ep=rollout(env,scene,expert,seed+attempt,'expert',limit=self.cfg['planner']['expert_limit'])
                        if not ep['success'][-1]: continue
                        length=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())
                        ratio=length/max(np.linalg.norm(goal-q0),1e-8)
                        gap=float(ep['clearance'].min())
                        direct=safe_edge(q0,goal,scene,self.cfg)
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
            raise RuntimeError(f'{split}: accepted {len(accepted)}/{count}; missing difficulty quotas: {missing}; rejection counts: {self.rejections}')
        finally:
            progress.close()
            env.close()
