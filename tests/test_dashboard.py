import json
from urllib.request import urlopen
from urllib.error import HTTPError
from http.server import ThreadingHTTPServer
from threading import Thread

import numpy as np
import pytest
import yaml

from minimal_recap.dashboard import Store, handler


@pytest.fixture
def store(tmp_path):
    run = tmp_path/'test_run'
    (run/'rollouts'/'round_00').mkdir(parents=True)
    (run/'labels'/'round_00').mkdir(parents=True)
    (run/'resolved.yaml').write_text(yaml.safe_dump({'observation': {'features': ['object', 'gripper', 'goal', 'pedestal']}}))
    meta = {'episode_id': 'bc_7', 'source': 'bc', 'status': 'complete', 'success': False}
    np.savez(run/'rollouts'/'round_00'/'000007.npz', metadata=json.dumps(meta),
             observation=np.zeros((2,25)), next_observation=np.ones((2,25)),
             reward=[-1.,-200.], action=np.zeros((2,4)), terminated=[False, True],
             truncated=[False,False], controller_states='[]')
    np.savez(run/'labels'/'round_00'/'bc_7.npz', value=[-190.,-200.], advantage=[-1.,0.])
    return Store(tmp_path)


def test_episode_mapping_terminal_and_label_link(store):
    summary = store.summary('test_run')
    assert summary['groups'] == [{'name': 'rollouts/round_00', 'count': 1}]
    listing = store.episodes('test_run', 'rollouts/round_00')
    data = store.episode('test_run', listing[0]['path'])
    assert data['columns'] == {'object': 0, 'gripper': 7, 'goal': 16, 'pedestal': 20}
    assert len(data['frames']) == 3
    assert data['frames'][-1] == [1.]*25
    assert data['returns'] == [-201., -200.]
    assert data['labels']['advantage'] == [-1.,0.]


@pytest.mark.parametrize('source,status,success,truncated', [
    ('bc', 'aborted', False, False),
    ('bc', 'complete', False, True),
    ('expert', 'complete', False, False),
])
def test_invalid_mc_is_not_displayed(store, source, status, success, truncated):
    path = store.root/'test_run'/'rollouts'/'round_00'/'000007.npz'
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    meta = json.loads(str(arrays['metadata']))
    meta.update(source=source, status=status, success=success)
    arrays['metadata'] = json.dumps(meta)
    arrays['truncated'] = [False, truncated]
    np.savez(path, **arrays)
    assert store.episode('test_run', 'rollouts/round_00/000007.npz')['returns'] is None


def test_http_read_only_and_path_boundary(store):
    server = ThreadingHTTPServer(('127.0.0.1',0), handler(store))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        assert json.load(urlopen(base+'/api/runs')) == ['test_run']
        assert b'RECAP' in urlopen(base+'/').read()
        with pytest.raises(HTTPError) as error:
            urlopen(base+'/api/episodes?run=test_run&group=../../')
        assert error.value.code == 400
        with pytest.raises(HTTPError) as error:
            urlopen(base+'/../dashboard.py')
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
