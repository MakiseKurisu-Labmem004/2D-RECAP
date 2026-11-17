import csv
import json

import numpy as np
import pytest
import torch

from minimal_recap.config import load, merge
from minimal_recap.learning import train
from minimal_recap.training_plots import export_training_curves


def test_initial_bc_budget():
    c = load('configs/experiments/initial_bc.yaml')
    s = c['bc_training']
    assert s['epochs'] * s['updates_per_epoch'] == 50000
    assert s['validation_every'] * s['updates_per_epoch'] == 2500
    assert s['validation_episodes'] == 100 and s['batch_size'] == 256
    assert c['run']['stages'] == ['physics', 'expert', 'freeze', 'bc', 'evaluate']


def test_interrupted_training_keeps_evaluation_schedule_and_curves(tmp_path):
    c = merge(load('configs/experiments/initial_bc.yaml'), {
        'model': {'device': 'cpu'},
        'bc_training': {'epochs': 4, 'updates_per_epoch': 2, 'batch_size': 4,
                        'validation_every': 2, 'log_progress': False},
    })
    rng = np.random.default_rng(42)
    eps = [{'observation': rng.normal(size=(8, 34)).astype(np.float32),
            'action': rng.uniform(-1, 1, size=(8, 4)).astype(np.float32),
            'metadata': {'episode_id': f'train_{i}', 'success': True}} for i in range(4)]
    output = tmp_path / 'interrupted' / 'checkpoints' / 'bc.pt'
    output.parent.mkdir(parents=True)
    calls = []

    def validate(model):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError('Simulated interruption during second evaluation')
        return {'episodes': 2, 'success_rate': 1., 'success_ci95': [0.34, 1.]}

    with pytest.raises(RuntimeError, match='Simulated interruption'):
        train(c, eps, output, 'bc', validate=validate)
    saved = torch.load(str(output)+'.training.pt', weights_only=False)
    assert saved['gradient_step'] == 6
    assert [r['gradient_step'] for r in saved['history']] == [2, 4, 6]
    resumed_calls = []

    def resumed_validation(model):
        resumed_calls.append(True)
        return {'episodes': 2, 'success_rate': 0., 'success_ci95': [0., 0.66]}

    train(c, eps, output, 'bc', validate=resumed_validation)
    assert len(resumed_calls) == 1
    history = json.loads((output.parent/'bc.pt.history.json').read_text())
    assert [r['gradient_step'] for r in history] == [2, 4, 6, 8]
    assert [r['gradient_step'] for r in history if 'id_validation' in r] == [4, 8]
    assert [r['id_validation']['success_rate'] for r in history if 'id_validation' in r] == [1., 0.]
    for row in history:
        if 'validation_checkpoint' in row:
            checkpoint = torch.load(output.parent/row['validation_checkpoint'], weights_only=False)
            assert checkpoint['metadata']['gradient_step'] == row['gradient_step']
            assert checkpoint['metadata']['id_validation'] == row['id_validation']
    # The selected model can precede the last step; each evaluated model survives.
    assert torch.load(output, weights_only=False)['metadata']['gradient_step'] == 4
    plot_dir = tmp_path/'interrupted/plots/bc'
    with (plot_dir/'curves.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert [r['gradient_step'] for r in rows] == ['2', '4', '6', '8']
    assert [r['id_success_rate'] for r in rows] == ['', '1.0', '', '0.0']
    assert (plot_dir/'loss.png').stat().st_size > 1000
    assert (plot_dir/'success_rate.png').stat().st_size > 1000

    # Check optimizer continuation against uninterrupted training, not only logs.
    uninterrupted = tmp_path/'uninterrupted/checkpoints/bc.pt'
    uninterrupted.parent.mkdir(parents=True)
    metrics = iter([history[1]['id_validation'], history[3]['id_validation']])
    train(c, eps, uninterrupted, 'bc', validate=lambda model: next(metrics))
    full = torch.load(str(uninterrupted)+'.training.pt', weights_only=False)
    resumed = torch.load(str(output)+'.training.pt', weights_only=False)
    for key, tensor in full['model'].items():
        torch.testing.assert_close(tensor, resumed['model'][key], rtol=0, atol=0)
    train(c, eps, output, 'bc', validate=lambda model: pytest.fail('Completed run evaluated again'))
    assert json.loads((output.parent/'bc.pt.history.json').read_text()) == history


def test_plot_without_evaluation_keeps_missing_values(tmp_path):
    export_training_curves([{'epoch': 1, 'train_loss': 0.2, 'validation_loss': 0.3}], tmp_path, 50)
    with (tmp_path/'curves.csv').open(newline='') as handle:
        row = next(csv.DictReader(handle))
    assert row['gradient_step'] == '50'
    assert row['id_success_rate'] == ''
