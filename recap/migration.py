"""Copy trajectory data to a new run, removing legacy joint velocity columns."""

import json
import math
from pathlib import Path
import shutil
from tempfile import mkdtemp

import numpy as np
from tqdm import tqdm

from .generation import write_json
from .observations import OBSERVATION_FIELDS, convert_episode


def migrate_run(source, destination, lr=None, train_overrides=None):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Migration requires separate source and destination directories')
    if destination.exists():
        raise FileExistsError(f'Destination already exists: {destination}')
    if lr is not None and (not math.isfinite(lr) or lr <= 0):
        raise ValueError('Learning rate must be finite and positive')
    cfg = json.loads((source / 'config.json').read_text(encoding='utf-8'))
    cfg['train'].pop('policy_use_velocity', None)
    if train_overrides is not None:
        if not isinstance(train_overrides, dict):
            raise ValueError('Training overrides must be a mapping')
        cfg['train'].update(train_overrides)
    if lr is not None:
        cfg['train']['lr'] = lr
    paths = sorted((source / 'data').rglob('*.npz'))
    if not paths:
        raise ValueError(f'No trajectory data found in {source / "data"}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    prefix = f'.{destination.name}.migrate-'
    temporary = Path(mkdtemp(prefix=prefix, dir=destination.parent)).resolve()
    try:
        with tqdm(paths, desc='Migrate observations', unit='episode', dynamic_ncols=True) as progress:
            for path in progress:
                with np.load(path, allow_pickle=False) as archive:
                    episode = convert_episode(dict(archive))
                output = temporary / path.relative_to(source)
                output.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(output, **episode)
        for filename in ('scenes.json', 'generation.json', 'difficulty.json'):
            if (source / filename).is_file():
                shutil.copyfile(source / filename, temporary / filename)
        write_json(temporary / 'config.json', cfg)
        write_json(temporary / 'migration.json', dict(source=str(source), episodes=len(paths),
                   observation_fields=OBSERVATION_FIELDS, learning_rate=cfg['train']['lr']))
        temporary.rename(destination)
    finally:
        # Only remove the staging directory created above, never either run.
        if temporary.exists():
            if temporary.parent != destination.parent or not temporary.name.startswith(prefix):
                raise RuntimeError('Unexpected migration staging directory')
            shutil.rmtree(temporary)
    tqdm.write(f'Migrated {len(paths)} episodes to {destination}; observations=9D, lr={cfg["train"]["lr"]}')
    return destination
