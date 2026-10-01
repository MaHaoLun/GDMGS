"""Build maximal solid cells from a triangle mesh and training camera centers."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'system'))
from occupancy import make_grid

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mesh', type=Path, required=True, help='NPZ with vertices/triangles, or mesh readable by trimesh')
    p.add_argument('--cameras', type=Path, required=True, help='JSON list of world-to-camera w2c matrices')
    p.add_argument('--level', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not 0 <= args.level <= 10:
        raise ValueError('level must be 0..10; dense grid needs O(8^level) memory, no automatic reduction')
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.mesh.suffix == '.npz':
        mesh = np.load(args.mesh, allow_pickle=False)
        vertices, faces = mesh['vertices'], mesh['triangles']
    else:
        import trimesh
        mesh = trimesh.load(args.mesh, process=False, force='mesh')
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    records = json.loads(args.cameras.read_text())
    eyes = np.array([np.linalg.solve(np.asarray(r['w2c'])[:3,:3], -np.asarray(r['w2c'])[:3,3]) for r in records])
    if not len(vertices) or not len(eyes) or not len(faces):
        raise ValueError('nonempty mesh and training-camera inventory required')
    result = make_grid(vertices, faces, eyes, args.level)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as f:
        np.savez_compressed(f, nodes=np.asarray(result.pop('full_nodes'), dtype=np.int64).reshape(-1,4),
                            lo=np.asarray(result['domain_lo']), hi=np.asarray(result['domain_hi']), level=args.level)
    args.output.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')

if __name__ == '__main__':
    main()
