import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('bc_search', Path(__file__).parents[1] / 'scripts' / 'search_bc.py')
search = importlib.util.module_from_spec(spec)
spec.loader.exec_module(search)


def test_ranking_uses_closed_loop_success_before_mse():
    high_success = dict(validation_success_rate=.8, validation_mse=.2)
    low_loss = dict(validation_success_rate=.7, validation_mse=.01)
    tied_success = dict(validation_success_rate=.8, validation_mse=.1)
    assert max([high_success, low_loss, tied_success], key=search.ranking) is tied_success


def test_completed_screen_stays_frozen_before_extended_checkpoint_selection(tmp_path):
    path = tmp_path / 'screen.json'
    rows = [dict(candidate='a', seed=42, best_step=2000), dict(candidate='b', seed=42, best_step=4000)]
    path.write_text(json.dumps(rows))
    assert search.completed_stage(path, ['a', 'b'], [42]) == rows
    assert search.completed_stage(path, ['a', 'b', 'c'], [42]) is None
    assert search.completed_stage(path, ['a', 'b'], [42, 43]) is None


def test_prepare_freezes_source_geometry_and_excludes_source_models_and_test_labels(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    cfg = dict(seed=42, device='auto', env=dict(horizon=500), train=dict(lr=.001, max_steps=8000))
    (source / 'config.json').write_text(json.dumps(cfg))
    (source / 'scenes.json').write_text(json.dumps(dict(train=['train'], validation=['val'], id=['test'])))
    for split in ('train', 'validation', 'id'):
        directory = source / 'data' / split / 'expert'
        directory.mkdir(parents=True)
        (directory / 'episode.npz').write_bytes(split.encode())
    (source / 'bc.pt').write_bytes(b'old model')
    before = search.fingerprint(source)
    monkeypatch.setattr(search, 'write', lambda p, obj: p.write_text(json.dumps(obj)))
    dest = tmp_path / 'experiment'
    search.prepare(source, dest, dict(lr=.0005), 43, 8000, 500, 'cpu')
    actual = search.read(dest / 'config.json')
    assert actual['env'] == cfg['env']
    assert actual['seed'] == 43 and actual['train']['lr'] == .0005
    assert search.read(dest / 'scenes.json') == search.read(source / 'scenes.json')
    assert (dest / 'data' / 'train' / 'expert' / 'episode.npz').exists()
    assert not (dest / 'data' / 'id').exists()
    assert not (dest / 'bc.pt').exists()
    search.prepare(source, dest, dict(lr=.0005), 43, 20000, 500, 'cpu')
    assert search.read(dest / 'config.json')['train']['max_steps'] == 20000
    with pytest.raises(ValueError, match='different settings'):
        search.prepare(source, dest, dict(lr=.001), 43, 20000, 500, 'cpu')
    assert search.fingerprint(source) == before
