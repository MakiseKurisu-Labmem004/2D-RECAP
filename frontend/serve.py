"""Local, dependency-free viewer for a RECAP run. Never modifies run files."""
import argparse
import ast
import json
import math
import struct
import zipfile
import threading
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


def read_npy(blob):
    if blob[:6] != b'\x93NUMPY':
        raise ValueError('Invalid NPY file')
    version = blob[6]
    offset = 10 if version == 1 else 12
    length = int.from_bytes(blob[8:offset], 'little')
    header = ast.literal_eval(blob[offset:offset + length].decode('latin1'))
    if header['fortran_order']:
        raise ValueError('Fortran-ordered arrays are unsupported')
    dtype, shape = header['descr'], header['shape']
    payload = blob[offset + length:]
    endian = '>' if dtype[0] == '>' else '<'
    kind, size = dtype[1], int(dtype[2:])
    if kind == 'U':
        width = size * 4
        values = [payload[i:i + width].decode('utf-32-be' if endian == '>' else 'utf-32-le').rstrip('\0') for i in range(0, len(payload), width)]
    else:
        code = {('f', 4): 'f', ('f', 8): 'd', ('i', 8): 'q', ('i', 4): 'i', ('b', 1): '?', ('u', 1): 'B'}.get((kind, size))
        if code is None:
            raise ValueError(f'Unsupported NPY dtype: {dtype}')
        values = [v[0] for v in struct.iter_unpack(endian + code, payload)]
        values = [round(v, 7) if isinstance(v, float) and math.isfinite(v) else None if isinstance(v, float) else v for v in values]
    if not shape:
        return values[0]
    for dimension in reversed(shape[1:]):
        values = [values[i:i + dimension] for i in range(0, len(values), dimension)]
    return values


def episode(path, fields):
    with zipfile.ZipFile(path) as archive:
        return {name: read_npy(archive.read(name + '.npy')) for name in fields if name + '.npy' in archive.namelist()}


def build_index(run):
    scenes = json.loads((run / 'scenes.json').read_text(encoding='utf-8'))
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    scene_ids = {scene['scene_id'] for group in scenes.values() for scene in group}
    rows = []
    for path in sorted((run / 'data').glob('*/*/*.npz')):
        item = episode(path, ['scene_id', 'episode_id', 'success', 'reward', 'failure_reason', 'policy_version'])
        if item['scene_id'] not in scene_ids:
            raise ValueError(f'{path}: scene_id missing from scenes.json')
        relative = path.relative_to(run / 'data')
        rows.append(dict(path=relative.as_posix(), scene_id=item['scene_id'], episode_id=item['episode_id'],
                         split=relative.parts[0], source=relative.parts[1], steps=len(item['reward']),
                         success=bool(item['success'][-1]), total_return=sum(item['reward']),
                         failure_reason=item['failure_reason'][-1]))
    generation_path = run / 'generation.json'
    difficulty_path = run / 'difficulty.json'
    order = {key: i for i, key in enumerate(scenes)}
    rows.sort(key=lambda row: (order[row['split']], row['source'], int(row['episode_id'].rsplit('-', 1)[-1]) if row['episode_id'].rsplit('-', 1)[-1].isdigit() else 0, row['path']))
    histories = {p.name.removesuffix('.history.json'): json.loads(p.read_text(encoding='utf-8')) for p in sorted(run.glob('*.history.json'))}
    evaluations = {}
    for path in sorted(run.glob('evaluation*.json')):
        evaluations.update(json.loads(path.read_text(encoding='utf-8')))
    migration_path = run / 'migration.json'
    return dict(run=run.name, config=config,
                histories=histories, evaluations=evaluations,
                migration=json.loads(migration_path.read_text(encoding='utf-8')) if migration_path.exists() else {},
                checkpoints=[p.name for p in sorted((run / 'checkpoints').glob('*.pt'))],
                generation=json.loads(generation_path.read_text(encoding='utf-8')) if generation_path.exists() else {},
                difficulty=json.loads(difficulty_path.read_text(encoding='utf-8')) if difficulty_path.exists() else {},
                refreshed_at=datetime.now(timezone.utc).isoformat(), scenes=scenes, episodes=rows)


class RunStore:
    """Refresh the index when manifests or episode files change."""
    def __init__(self, run):
        self.run, self.signature, self.index = run, None, None
        self.lock = threading.Lock()

    def get(self):
        with self.lock:
            files = [self.run / name for name in ('config.json', 'scenes.json', 'generation.json', 'difficulty.json')]
            files.extend(sorted((self.run / 'data').glob('*/*/*.npz')))
            files.extend(sorted(self.run.glob('*.history.json')))
            files.extend(sorted(self.run.glob('evaluation*.json')))
            files.extend(sorted((self.run / 'checkpoints').glob('*.pt')))
            files.append(self.run / 'migration.json')
            signature = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files if p.exists())
            if signature != self.signature:
                index = build_index(self.run)
                self.index, self.signature = index, signature
            return self.index


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, run, store, **kwargs):
        self.run, self.store = run, store
        super().__init__(*args, directory=str(Path(__file__).parent), **kwargs)

    def send_json(self, value):
        data = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlsplit(self.path)
        try:
            if url.path == '/api/data':
                return self.send_json(self.store.get())
            if url.path == '/api/episode':
                relative = parse_qs(url.query).get('path', [''])[0]
                if relative not in {e['path'] for e in self.store.get()['episodes']}:
                    return self.send_error(404)
                return self.send_json(episode(self.run / 'data' / relative,
                    ['qpos', 'action', 'reward', 'clearance', 'success', 'failure_reason', 'scene_id', 'episode_id']))
            if url.path not in ('/', '/index.html', '/app.js', '/research.js', '/style.css', '/favicon.svg'):
                return self.send_error(404)
            return super().do_GET()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as error:
            self.log_error('%s', error)
            self.send_error(500, 'Unable to read experiment data')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path(__file__).resolve().parent.parent / 'runs' / 'seed42-8k')
    parser.add_argument('--port', type=int, default=8042)
    args = parser.parse_args()
    run = args.run.resolve()
    store = RunStore(run)
    index = store.get()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(Handler, run=run, store=store))
    print(f'RECAP viewer: http://127.0.0.1:{args.port} | {len(index["episodes"])} episodes', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
