"""Load a real checkpoint, select, share materialization, and render all targets."""
import argparse
import json
from pathlib import Path
from system.bootstrap import configure

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify', action='store_true', help='check every selection, group decode and output')
    args = parser.parse_args()
    configure()
    from full_system import run
    run(json.loads(args.config.read_text()), args.output, args.verify)
