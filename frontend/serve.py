"""Local, dependency-free viewer for a RECAP run. Never modifies run files."""
import argparse
import ast
import json
import math
import struct
import zipfile
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
    rows = []
    for path in sorted((run / 'data').glob('*/*/*.npz')):
        item = episode(path, ['scene_id', 'episode_id', 'success', 'reward', 'failure_reason', 'policy_version'])
        relative = path.relative_to(run / 'data')
        rows.append(dict(path=relative.as_posix(), scene_id=item['scene_id'], episode_id=item['episode_id'],
                         split=relative.parts[0], source=relative.parts[1], steps=len(item['reward']),
                         success=bool(item['success'][-1]), total_return=sum(item['reward']),
                         failure_reason=item['failure_reason'][-1]))
    generation_path = run / 'generation.json'
    return dict(run=run.name, config=json.loads((run / 'config.json').read_text(encoding='utf-8')),
                generation=json.loads(generation_path.read_text(encoding='utf-8')) if generation_path.exists() else {},
                scenes=scenes, episodes=rows)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, run, index, **kwargs):
        self.run, self.index = run, index
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
                return self.send_json(self.index)
            if url.path == '/api/episode':
                relative = parse_qs(url.query).get('path', [''])[0]
                if relative not in {e['path'] for e in self.index['episodes']}:
                    return self.send_error(404)
                return self.send_json(episode(self.run / 'data' / relative,
                    ['qpos', 'action', 'reward', 'clearance', 'success', 'failure_reason', 'scene_id', 'episode_id']))
            if url.path not in ('/', '/index.html', '/app.js', '/style.css', '/favicon.svg'):
                return self.send_error(404)
            return super().do_GET()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as error:
            self.log_error('%s', error)
            self.send_error(500, 'Unable to read experiment data')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path(__file__).resolve().parent.parent / 'runs' / 'seed42')
    parser.add_argument('--port', type=int, default=8042)
    args = parser.parse_args()
    run = args.run.resolve()
    index = build_index(run)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(Handler, run=run, index=index))
    print(f'RECAP viewer: http://127.0.0.1:{args.port} | {len(index["episodes"])} episodes', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
