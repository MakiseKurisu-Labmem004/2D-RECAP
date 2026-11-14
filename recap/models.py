import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def returns(rewards,scale):
    return np.cumsum(np.asarray(rewards,dtype=np.float64)[::-1])[::-1].copy()/scale


def advantage(rewards,values,next_values,n,scale):
    # Episodes are processed separately; timeout is terminal for this finite-horizon return.
    out=np.zeros(len(rewards))
    for t in range(len(rewards)):
        end=min(t+n,len(rewards))
        out[t]=sum(rewards[t:end])/scale-values[t]
        if end<len(rewards): out[t]+=next_values[end-1]
    return out


def distribution_targets(y,bins):
    x=y.clamp(-1,0).add(1).mul(bins-1)
    lo=x.floor().long(); hi=(lo+1).clamp(max=bins-1)
    target=torch.zeros(len(y),bins,device=y.device)
    target.scatter_add_(1,lo[:,None],(1-(x-lo))[:,None])
    target.scatter_add_(1,hi[:,None],(x-lo)[:,None])
    return target


class Model(nn.Module):
    def __init__(self,kind,cfg,limit):
        super().__init__(); self.kind=kind; self.limit=limit; self.scalar=cfg['scalar_value']
        size=14 if kind=='recap' else 11
        layers=[]
        for _ in range(cfg['depth']):
            layers += [nn.Linear(size,cfg['width']),nn.SiLU()]; size=cfg['width']
        output=4 if kind=='recap' else (1 if self.scalar else cfg['bins']) if kind=='value' else 2
        self.net=nn.Sequential(*layers,nn.Linear(size,output))
        self.register_buffer('support',torch.linspace(-1,0,cfg['bins']))
    def forward(self,obs,condition=None):
        if self.kind=='recap':
            if condition is None: condition=torch.full((len(obs),),2,device=obs.device,dtype=torch.long)
            obs=torch.cat([obs,F.one_hot(condition,3).float()],-1)
        out=self.net(obs)
        if self.kind in ('bc','sft'): return out.tanh()*self.limit
        return out
    def value(self,obs):
        out=self(obs)
        return -out[:,0].sigmoid() if self.scalar else (out.softmax(-1)*self.support).sum(-1)
    def nll(self,obs,actions,condition):
        mean,logstd=self(obs,condition).chunk(2,-1)
        logstd=logstd.clamp(-5,1)
        x=(actions/self.limit).clamp(-1+1e-6,1-1e-6); z=torch.atanh(x)
        return -(torch.distributions.Normal(mean,logstd.exp()).log_prob(z)-torch.log(self.limit*(1-x.square())+1e-8)).sum(-1)
    def action(self,obs):
        if self.kind!='recap': return self(obs)
        # Deterministic evaluation uses the squashed Gaussian location.
        return self(obs,torch.ones(len(obs),device=obs.device,dtype=torch.long))[:,:2].tanh()*self.limit
