"""Read-only local experiment dashboard: python -m minimal_recap.dashboard."""
import argparse
import json
import mimetypes
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
import yaml

STATIC = Path(__file__).with_name('dashboard_web')


def inside(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Path outside runs directory')
    return path


def read_json(path, fallback=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else fallback


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.cache = {}

    def run_path(self, name):
        path = inside(self.root, name)
        if path.parent != self.root or not path.is_dir():
            raise FileNotFoundError('Run not found')
        return path

    def groups(self, run):
        groups = {}
        for p in run.rglob('*.npz'):
            if p.is_symlink() or not p.resolve().is_relative_to(run):
                continue
            group = p.parent.relative_to(run).as_posix()
            if group == 'labels' or group.startswith('labels/'):
                continue
            groups[group] = groups.get(group, 0) + 1
        return [{'name': k, 'count': v} for k, v in sorted(groups.items())]

    def summary(self, name):
        run = self.run_path(name)
        config = yaml.safe_load((run / 'resolved.yaml').read_text(encoding='utf-8')) if (run / 'resolved.yaml').exists() else {}
        histories = {p.name.replace('.pt.history.json', ''): read_json(p) for p in sorted((run / 'checkpoints').glob('*.history.json'))}
        label_groups = ['rollouts/'+p.name for p in sorted((run/'labels').glob('round_*'))
                        if p.is_dir() and any(p.glob('*.npz')) and (run/'rollouts'/p.name).is_dir()]
        label_count = sum(1 for p in (run/'labels').rglob('*.npz') if p.is_file())
        return dict(name=name, groups=self.groups(run), label_groups=label_groups, label_count=label_count, metrics=read_json(run/'metrics.json', {}),
                    histories=histories, config=config, manifest=read_json(run/'manifest.json', {}),
                    contract=read_json(run/'contract.json', {}), stages=[p.name.removesuffix('.done.json') for p in run.glob('*.done.json')])

    def episodes(self, name, group):
        run = self.run_path(name)
        folder = run if group == '__all__' else inside(run, group)
        result = []
        paths = folder.rglob('*.npz') if group == '__all__' else folder.glob('*.npz')
        for p in sorted(paths):
            if not p.resolve().is_relative_to(run):
                continue
            relative = p.relative_to(run)
            if relative.parts[0] == 'labels':
                continue
            stat = p.stat()
            key = str(p), stat.st_mtime_ns, stat.st_size
            if key not in self.cache:
                with np.load(p, allow_pickle=False) as f:
                    if 'metadata' not in f:
                        continue
                    meta = json.loads(str(f['metadata']))
                    self.cache[key] = dict(path=p.relative_to(run).as_posix(), metadata=meta,
                                           length=len(f['reward']), total_return=float(f['reward'].sum()))
            row = dict(self.cache[key])
            row['group'] = p.parent.relative_to(run).as_posix()
            row['has_labels'] = (relative.parts[0] == 'rollouts' and
                                 (run/'labels'/p.parent.name/(row['metadata']['episode_id']+'.npz')).is_file())
            result.append(row)
        return result

    def episode(self, name, path):
        run = self.run_path(name)
        p = inside(run, path)
        if p.suffix != '.npz':
            raise ValueError('Expected NPZ episode')
        with np.load(p, allow_pickle=False) as f:
            meta = json.loads(str(f['metadata']))
            obs, nxt = f['observation'], f['next_observation']
            reward = f['reward']
            frames = np.concatenate([obs, nxt[-1:]], axis=0) if len(nxt) else obs
            config = yaml.safe_load((run/'resolved.yaml').read_text(encoding='utf-8'))
            sizes = dict(gripper=9, object=7, goal=4, pedestal=5, relative=6, geometry=3, arm=6)
            columns, offset = {}, 0
            for feature in config['observation']['features']:
                columns[feature] = offset
                offset += sizes[feature]
            if obs.ndim != 2 or obs.shape[1] != offset:
                raise ValueError('Observation width does not match resolved.yaml features')
            complete = meta.get('status') == 'complete' and len(reward) and bool(f['terminated'][-1]) and not f['truncated'].any()
            valid_return = bool(complete and not (meta.get('source') == 'expert' and not meta.get('success')))
            result = dict(metadata=meta, frames=frames, columns=columns, action=f['action'], reward=reward,
                          returns=np.cumsum(reward[::-1])[::-1] if valid_return else None,
                          phases=[s.get('stage') for s in json.loads(str(f['controller_states']))], labels={})
        # Label files are keyed by episode ID, not by the numeric NPZ filename.
        if path.startswith('rollouts/'):
            label = inside(run, 'labels/'+p.parent.name+'/'+meta['episode_id']+'.npz')
            if label.exists():
                with np.load(label, allow_pickle=False) as f:
                    result['labels'] = {k: f[k] for k in f.files}
                result['label_info'] = read_json(label.parent/'manifest.json', {}).get(meta['episode_id'], {})
        return clean(result)


def handler(store):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                url = urlparse(self.path)
                q = parse_qs(url.query)
                arg = lambda k: q[k][0]
                if url.path == '/api/runs':
                    value = [p.name for p in sorted(store.root.iterdir()) if p.is_dir() and not p.is_symlink()]
                elif url.path == '/api/run':
                    value = store.summary(arg('run'))
                elif url.path == '/api/episodes':
                    value = store.episodes(arg('run'), arg('group'))
                elif url.path == '/api/episode':
                    value = store.episode(arg('run'), arg('path'))
                elif url.path.startswith('/api/'):
                    self.send_error(404)
                    return
                else:
                    p = inside(STATIC, url.path.lstrip('/') or 'index.html')
                    data = p.read_bytes()
                    self.send_response(200)
                    self.send_header('Content-Type', mimetypes.guess_type(str(p))[0] or 'application/octet-stream')
                    self.send_header('Content-Length', str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                data = json.dumps(clean(value), ensure_ascii=False, allow_nan=False).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (ValueError, KeyError) as ex:
                self.send_error(400, str(ex))
            except FileNotFoundError:
                self.send_error(404, 'Data not found')
            except Exception:
                self.log_error('Unable to read %s', self.path)
                self.send_error(500, 'Unable to read experiment data; check the file format')

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=Path('runs'))
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    if not args.runs.is_dir():
        parser.error('Runs directory does not exist')
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler(Store(args.runs)))
    print(f'RECAP dashboard: http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
