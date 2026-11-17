"""Headless training curves: python -m minimal_recap.training_plots --run runs/initial_bc_50k_seed1."""
import argparse
import csv
import json
from pathlib import Path

import yaml


def export_training_curves(history, output, updates_per_epoch):
    """Export recorded metrics only; missing policy evaluations stay missing."""
    if not history:
        raise ValueError('No training history to plot')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in history:
        metrics = item.get('id_validation', {})
        interval = metrics.get('success_ci95', [None, None])
        rows.append({
            'gradient_step': item.get('gradient_step', item['epoch'] * updates_per_epoch),
            'epoch': item['epoch'],
            'train_loss': item['train_loss'],
            'validation_loss': item['validation_loss'],
            'id_success_rate': metrics.get('success_rate'),
            'id_success_ci95_low': interval[0],
            'id_success_ci95_high': interval[1],
            'id_episodes': metrics.get('episodes'),
            'validation_checkpoint': item.get('validation_checkpoint'),
        })
    csv_path = output / 'curves.csv'
    temporary = csv_path.with_suffix('.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)

    def save(fig, name):
        path = output / name
        temporary = path.with_suffix('.tmp.png')
        try:
            fig.tight_layout()
            fig.savefig(temporary, dpi=160)
            temporary.replace(path)
        finally:
            plt.close(fig)

    steps = [r['gradient_step'] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, [r['train_loss'] for r in rows], label='Train (update-block mean)', color='#237c68')
    ax.plot(steps, [r['validation_loss'] for r in rows], label='Held-out demonstration loss', color='#507ca9')
    ax.set(xlabel='Gradient step', ylabel='Loss', title='Training and validation loss')
    ax.grid(alpha=0.2)
    ax.legend()
    save(fig, 'loss.png')

    evaluated = [r for r in rows if r['id_success_rate'] is not None]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if evaluated:
        x = [r['gradient_step'] for r in evaluated]
        y = [100 * r['id_success_rate'] for r in evaluated]
        ax.plot(x, y, 'o-', color='#237c68', label='ID validation success')
        intervals = [r for r in evaluated if r['id_success_ci95_low'] is not None and r['id_success_ci95_high'] is not None]
        if intervals:
            ax.fill_between([r['gradient_step'] for r in intervals],
                            [100 * r['id_success_ci95_low'] for r in intervals],
                            [100 * r['id_success_ci95_high'] for r in intervals],
                            color='#237c68', alpha=0.15, label='95% Wilson interval')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No policy evaluation recorded yet', ha='center', transform=ax.transAxes)
    ax.set(xlabel='Gradient step', ylabel='Success rate (%)', ylim=(0, 100), title='Fixed ID validation scenes')
    ax.grid(alpha=0.2)
    save(fig, 'success_rate.png')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--model', default='bc', help='Checkpoint stem, e.g. bc or conditioned_00')
    args = parser.parse_args()
    config = yaml.safe_load((args.run / 'resolved.yaml').read_text(encoding='utf-8'))
    history = json.loads((args.run / 'checkpoints' / f'{args.model}.pt.history.json').read_text(encoding='utf-8'))
    section = 'bc_training' if args.model == 'bc' else 'value_training' if args.model.startswith('value_') else 'recap'
    print(export_training_curves(history, args.run / 'plots' / args.model, config[section]['updates_per_epoch']).resolve())


if __name__ == '__main__':
    main()
