from pathlib import Path
import numpy as np
import torch
from torch import nn
from .data import returns, split, atomic_json
from .config import seed_all, validate_update_method


def mlp(dim, out, c):
    activation = {'relu': nn.ReLU, 'tanh': nn.Tanh}[c['activation']]
    layers = []
    for _ in range(c['hidden_layers']):
        layers += [nn.Linear(dim, c['hidden_units']), activation()]
        dim = c['hidden_units']
    return nn.Sequential(*layers, nn.Linear(dim, out))


class Network(nn.Module):
    def __init__(self, dim, c, kind='policy', conditional=False, support=None, action_dim=4):
        super().__init__()
        self.spec = dict(dim=dim, c=c, kind=kind, conditional=conditional, support=support, action_dim=action_dim)
        self.kind, self.conditional = kind, conditional
        self.net = mlp(dim+int(conditional), action_dim if kind == 'policy' else c['value_bins'], c)
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('scale', torch.ones(dim))
        self.register_buffer('support', torch.tensor([] if support is None else np.linspace(*support, c['value_bins']), dtype=torch.float32))

    def forward(self, x, condition=None):
        x = (x.to(self.mean.device)-self.mean)/self.scale
        if self.conditional:
            if condition is None:
                condition = torch.ones((*x.shape[:-1], 1), device=x.device)
            x = torch.cat([x, condition.to(x.device)], -1)
        output = self.net(x)
        return torch.tanh(output) if self.kind == 'policy' else output

    @torch.no_grad()
    def value(self, x):
        return (self(x).softmax(-1)*self.support).sum(-1)

    def normalize(self, obs, enabled):
        if enabled:
            self.mean.copy_(torch.as_tensor(obs.mean(0), dtype=torch.float32))
            self.scale.copy_(torch.as_tensor(np.maximum(obs.std(0), 1e-3), dtype=torch.float32))


def save_model(path, model, metadata):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    torch.save({'spec': model.spec, 'state': model.state_dict(), 'metadata': metadata}, temp)
    temp.replace(path)


def load_model(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model = Network(**ck['spec'])
    model.load_state_dict(ck['state'])
    return model.eval()


class Behavior:
    def __init__(self, model, exploration, seed):
        self.model = model.eval()
        self.exploration = exploration
        self.rng = np.random.default_rng(seed)

    def restore_rng(self, state):
        self.rng.bit_generator.state = state

    @torch.no_grad()
    def __call__(self, obs):
        a = self.model(torch.tensor(obs, dtype=torch.float32)).cpu().numpy()
        mode = self.exploration['mode']
        if mode == 'gaussian':
            a += self.rng.normal(0, self.exploration['std'], a.shape)
        elif mode != 'none':
            raise ValueError('Unknown exploration rule')
        return np.clip(a, -1, 1).astype(np.float32)


def bin_targets(y, support, soft=True):
    if not torch.isfinite(y).all() or (y < support[0]).any() or (y > support[-1]).any():
        raise ValueError(f'Return outside frozen support [{support[0]}, {support[-1]}]')
    index = (y-support[0])/(support[-1]-support[0])*(len(support)-1)
    low = index.floor().long()
    high = (low+1).clamp(max=len(support)-1)
    if not soft:
        return nn.functional.one_hot(index.round().long(), len(support)).float()
    weight = index-low
    out = torch.zeros((len(y), len(support)), dtype=torch.float32)
    out.scatter_add_(1, low[:, None], (1-weight)[:, None])
    out.scatter_add_(1, high[:, None], weight[:, None])
    return out


def advantages(ep, values, next_values, n):
    if n < 1:
        raise ValueError('n must be >= 1')
    returns(ep)  # require complete trajectories
    r, term = np.asarray(ep['reward']), np.asarray(ep['terminated'])
    out = np.empty(len(r), dtype=np.float32)
    for t in range(len(r)):
        end = min(t+n, len(r))
        terminals = np.flatnonzero(term[t:end])
        if len(terminals):
            end = t+int(terminals[0])+1
        bootstrap = 0 if term[end-1] else next_values[end-1]
        out[t] = r[t:end].sum()+bootstrap-values[t]
    return out


def warm_start(model, initial):
    state = initial.state_dict()
    if model.conditional and not initial.conditional:
        state['net.0.weight'] = torch.cat([state['net.0.weight'], torch.zeros((state['net.0.weight'].shape[0], 1))], 1)
    model.load_state_dict(state)


def train(c, eps, output, kind, contract=None, initial=None, experts=None, labels=None, validate=None):
    if kind == 'update':
        validate_update_method(c['recap']['method'])
    if kind == 'value' and any(e['metadata'].get('source') == 'expert' for e in eps):
        raise ValueError('Unbounded expert returns are not finite-budget value targets')
    seed_all(c['seeds']['training'])
    section = c['bc_training'] if kind == 'bc' else c['value_training'] if kind == 'value' else c['recap']
    method = c['recap']['method']
    conditional = kind == 'update' and method == 'conditioned'
    if kind == 'update' and method == 'success_bc':
        eps = [e for e in eps if e['metadata']['success']]
    tr, va = split(eps, c['dataset']['validation_fraction'], c['seeds']['training'])
    def stack(items, key):
        return np.concatenate([e[key] for e in items])
    obs, val_obs = stack(tr, 'observation'), stack(va, 'observation')
    model = Network(obs.shape[1], c['model'], 'value' if kind == 'value' else 'policy', conditional,
                    contract['support'] if kind == 'value' else None, action_dim=stack(tr, 'action').shape[1])
    model.normalize(obs, c['observation']['normalize'])
    if initial is not None:
        # Keep BC statistics fixed when warm starting policy updates.
        warm_start(model, initial)
    X, VX = torch.tensor(obs, dtype=torch.float32), torch.tensor(val_obs, dtype=torch.float32)
    if kind == 'value':
        Y = bin_targets(torch.tensor(np.concatenate([returns(e) for e in tr]), dtype=torch.float32), model.support, c['model']['soft_bins'])
        VY = bin_targets(torch.tensor(np.concatenate([returns(e) for e in va]), dtype=torch.float32), model.support, c['model']['soft_bins'])
    else:
        Y, VY = torch.tensor(stack(tr, 'action'), dtype=torch.float32), torch.tensor(stack(va, 'action'), dtype=torch.float32)
    def conditions(items):
        return torch.tensor(np.concatenate([labels[e['metadata']['episode_id']] for e in items]) > c['recap']['threshold'], dtype=torch.float32)[:, None]
    C, VC = (conditions(tr), conditions(va)) if conditional else (None, None)
    expert_X = expert_Y = None
    if kind == 'update' and experts and c['recap']['expert_ratio'] > 0:
        # Whole-trajectory split prevents expert holdout leakage.
        expert_train, _ = split(experts, c['dataset']['validation_fraction'], c['seeds']['training'])
        expert_X = torch.tensor(stack(expert_train, 'observation'), dtype=torch.float32)
        expert_Y = torch.tensor(stack(expert_train, 'action'), dtype=torch.float32)
    model.to(c['model'].get('device', 'cpu'))
    optimizer = torch.optim.Adam(model.parameters(), lr=section['lr'])
    rng = np.random.default_rng(c['seeds']['training'])
    best, history, start_epoch = float('inf'), [], 0
    resume_path = Path(str(output)+'.training.pt')
    if resume_path.exists():
        resume = torch.load(resume_path, map_location='cpu', weights_only=False)
        model.load_state_dict(resume['model'])
        optimizer.load_state_dict(resume['optimizer'])
        rng.bit_generator.state = resume['rng']
        torch.set_rng_state(resume['torch_rng'])
        best, history, start_epoch = resume['best'], resume['history'], resume['epoch']
    def loss(x, y, condition):
        pred = model(x, condition)
        y = y.to(pred.device)
        if kind == 'value':
            return -(y*pred.log_softmax(-1)).sum(-1).mean()
        per = (pred-y).square().mean(-1)
        if condition is not None:
            per = per*torch.where(condition[:, 0].to(pred.device) > 0, c['recap']['condition_weight'], 1.)
        return per.mean()
    for epoch in range(start_epoch, section['epochs']):
        model.train()
        losses = []
        for _ in range(section['updates_per_epoch']):
            ids = rng.integers(len(X), size=section['batch_size'])
            optimizer.zero_grad()
            objective = loss(X[ids], Y[ids], C[ids] if C is not None else None)
            if expert_X is not None:
                ratio = c['recap']['expert_ratio']
                ei = rng.integers(len(expert_X), size=len(ids))
                ex_loss = loss(expert_X[ei], expert_Y[ei], torch.ones((len(ids), 1)) if conditional else None)
                objective = (1-ratio)*objective+ratio*ex_loss
            objective.backward()
            optimizer.step()
            losses.append(float(objective.detach()))
        model.eval()
        with torch.no_grad():
            val_loss = float(loss(VX, VY, VC))
        score = val_loss
        gradient_step = (epoch+1)*section['updates_per_epoch']
        row = {'epoch': epoch+1, 'gradient_step': gradient_step,
               'train_loss': float(np.mean(losses)), 'validation_loss': val_loss}
        eligible = True
        if validate:
            eligible = (epoch+1) % section.get('validation_every', 1) == 0 or epoch+1 == section['epochs']
            if eligible:
                if section.get('log_progress', False):
                    print(f'{kind} step={gradient_step}: evaluating ID validation scenes', flush=True)
                metrics = validate(model)
                row['id_validation'] = metrics
                score = -metrics['success_rate'] + val_loss*1e-5
        metadata = {'kind': kind, 'epoch': epoch+1, 'gradient_step': gradient_step, 'contract': contract,
                    'training_episodes': [e['metadata']['episode_id'] for e in tr],
                    'validation_episodes': [e['metadata']['episode_id'] for e in va],
                    'mixing': c['dataset']['mixing'], 'method': method if kind == 'update' else kind}
        if 'id_validation' in row:
            metadata['id_validation'] = row['id_validation']
            if section.get('save_validation_checkpoints', False):
                checkpoint = Path(output).parent / f'{Path(output).stem}_steps' / f'step_{gradient_step:06}.pt'
                checkpoint.parent.mkdir(exist_ok=True)
                save_model(checkpoint, model, metadata)
                row['validation_checkpoint'] = checkpoint.relative_to(Path(output).parent).as_posix()
        history.append(row)
        if eligible and score < best:
            best = score
            save_model(output, model, metadata)
        temporary = resume_path.with_suffix('.tmp')
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'rng': rng.bit_generator.state,
                    'torch_rng': torch.get_rng_state(), 'best': best, 'history': history,
                    'epoch': epoch+1, 'gradient_step': gradient_step}, temporary)
        temporary.replace(resume_path)
        atomic_json(str(output)+'.history.json', history)
        if section.get('log_progress', False):
            success = f' id_success={row["id_validation"]["success_rate"]:.1%}' if 'id_validation' in row else ''
            print(f'{kind} step={gradient_step}/{section["epochs"]*section["updates_per_epoch"]}'
                  f' train_loss={row["train_loss"]:.6f} validation_loss={val_loss:.6f}{success}', flush=True)
        if section.get('export_curves', False) and 'id_validation' in row:
            from .training_plots import export_training_curves
            export_training_curves(history, Path(output).parent.parent/'plots'/Path(output).stem, section['updates_per_epoch'])
    # Also reconstruct plots when resuming an already-completed training file.
    if section.get('export_curves', False):
        from .training_plots import export_training_curves
        export_training_curves(history, Path(output).parent.parent/'plots'/Path(output).stem, section['updates_per_epoch'])
    return load_model(output)
