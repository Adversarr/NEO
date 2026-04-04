import trimesh
from argparse import ArgumentParser
import numpy as np
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--subdivisions", type=int, default=1)
    return parser.parse_args()

def main(args):
    mesh: trimesh.Trimesh = trimesh.load(args.input)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    print(f"Subdivided mesh with {args.subdivisions} iterations.")
    print(f"Input: #v={mesh.vertices.shape[0]}, #f={mesh.faces.shape[0]}")
    if args.subdivisions > 0:
        try:
            mesh = mesh.subdivide_loop(iterations=args.subdivisions)
        except Exception:
            mesh = mesh.subdivide(iterations=args.subdivisions)
    print(f"Output: #v={mesh.vertices.shape[0]}, #f={mesh.faces.shape[0]}")
    mesh.export(args.output)
    print(f"✅ Subdivided mesh saved to {args.output}")


if __name__ == "__main__":
    args = parse_args()
    main(args)