"""Re-execute smooth experts on existing scenes without resampling or overwriting data."""

import argparse
from pathlib import Path
import yaml

from recap.reexpert import rebuild_experts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default='configs/default.yaml', help='New expert and training settings; source environment stays fixed')
    args = parser.parse_args()
    settings = yaml.safe_load(Path(args.config).read_text())
    result = rebuild_experts(args.run, args.output, settings)
    print(f'Smooth experts saved to {result}')


if __name__ == '__main__':
    main()
