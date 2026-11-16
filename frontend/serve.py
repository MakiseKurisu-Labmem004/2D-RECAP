"""Local, dependency-free viewer for a RECAP run. Never modifies run files."""
import argparse
import hashlib
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
    scene_map = {scene['scene_id']: scene for group in scenes.values() for scene in group}
    rows = []
    collections = [dict(id='base', label='基础数据 · Expert / BC')]
    improvements = []
    paths = [(p, 'base') for p in sorted((run / 'data').glob('*/*/*.npz'))]
    def read_json(path, default):
        return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default
    def metrics(report):
        return {key: value for key, value in ((report or {}).get('metrics') or {}).items() if not isinstance(value, (list, dict))}
    for experiment in sorted((run / 'improvement').glob('*')):
        if not experiment.is_dir():
            continue
        protocol = read_json(experiment / 'protocol.json', {})
        baseline = read_json(experiment / 'baseline.json', {})
        study = dict(name=experiment.name, protocol={key: protocol.get(key) for key in ('evaluation_protocol', 'scene_pool_size', 'scene_pool_sources', 'options')},
                     holdout_scenes=len(protocol.get('holdout_scene_ids', [])), baseline={key: metrics(report) for key, report in baseline.items()}, rounds=[])
        for mode in protocol.get('modes', ['autonomous', 'corrections']):
            for round_path in sorted((experiment / mode).glob('round-*')):
                if not round_path.is_dir():
                    continue
                context = round_path.relative_to(run).as_posix()
                label = f'{experiment.name} / {mode} / {round_path.name}'
                collections.append(dict(id=context, label=label))
                paths.extend((p, context) for p in sorted((round_path / 'data').glob('*/*/*.npz')))
                result = read_json(round_path / 'result.json', {})
                gate = result.get('gate', {})
                study['rounds'].append(dict(collection=context, mode=mode, round=result.get('round', round_path.name),
                    budget=result.get('budget', {}), accepted=gate.get('accepted'),
                    recovered=len(gate.get('recovered', [])), regressions=len(gate.get('regressions', [])),
                    validation_regressions=len(gate.get('validation_regressions', [])),
                    candidate=metrics(result.get('candidate_adaptation')), active=metrics(result.get('active_adaptation')),
                    validation=metrics(result.get('candidate_validation')),
                    diagnostics=read_json(round_path / 'diagnostics.json', {}),
                    histories={p.name.removesuffix('.history.json'): [{k: v for k, v in row.items() if not isinstance(v, (list, dict))} for row in read_json(p, [])] for p in round_path.glob('*.history.json')}))
        improvements.append(study)
    for path, context in paths:
        item = episode(path, ['scene_id', 'episode_id', 'success', 'reward', 'failure_reason', 'policy_version'])
        if item['scene_id'] not in scene_map:
            raise ValueError(f'{path}: scene_id missing from scenes.json')
        rows.append(dict(path=path.relative_to(run).as_posix(), collection=context, scene_id=item['scene_id'], episode_id=item['episode_id'],
                         split=scene_map[item['scene_id']]['split'], source=path.parent.name, steps=len(item['reward']),
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
    spec_path = run / 'data.spec.json'
    return dict(run=run.name, config=config, collections=collections, improvements=improvements,
                histories=histories, evaluations=evaluations,
                data_spec=json.loads(spec_path.read_text(encoding='utf-8')) if spec_path.exists() else {},
                training_configs={p.name.removeprefix('training.').removesuffix('.json'): json.loads(p.read_text(encoding='utf-8')) for p in sorted(run.glob('training.*.json'))},
                goal_anchors={p.name.removesuffix('.goal_anchors.json'): json.loads(p.read_text(encoding='utf-8')) for p in sorted(run.glob('*.goal_anchors.json'))},
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
            files.append(self.run / 'data.spec.json')
            files.extend(sorted(self.run.glob('training.*.json')))
            files.extend(sorted(self.run.glob('*.goal_anchors.json')))
            files.extend(sorted((self.run / 'improvement').rglob('*.json')))
            files.extend(sorted((self.run / 'improvement').rglob('*.npz')))
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
                raw = episode(self.run / relative,
                    ['qpos', 'action', 'reward', 'clearance', 'success', 'failure_reason', 'scene_id', 'episode_id',
                     'distance', 'goal_hold_steps', 'success_distance', 'strict_success_distance',
                     'goal_hold_required', 'control_dt', 'physics_timestep', 'executed_substeps',
                     'initial_qpos', 'scene_json', 'takeover_step', 'control_step_offset', 'intervention_mask', 'task_success', 'parent_episode_id', 'value_prediction', 'mc_return', 'td_advantage',
                     'advantage_indicator', 'value_alignment', 'advantage_threshold', 'n_step', 'return_scale',
                     'value_checkpoint', 'value_checkpoint_sha256', 'annotation_round', 'within_task_horizon'])
                cache = Path(__file__).resolve().parent / '.cache' / 'value' / self.run.name / Path(relative).with_suffix('.json')
                if not all(key in raw for key in ('value_prediction', 'td_advantage')) and cache.exists():
                    saved = json.loads(cache.read_text(encoding='utf-8'))
                    critic = (self.run / saved['value_checkpoint']).resolve()
                    if critic.is_relative_to(self.run.resolve()) and critic.is_file() and saved['trajectory_sha256'] == hashlib.sha256((self.run / relative).read_bytes()).hexdigest() and saved['value_checkpoint_sha256'] == hashlib.sha256(critic.read_bytes()).hexdigest():
                        raw.update(saved)
                return self.send_json(raw)
            if url.path not in ('/', '/index.html', '/app.js', '/research.js', '/improvement.js', '/learning.js', '/style.css', '/favicon.svg'):
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
