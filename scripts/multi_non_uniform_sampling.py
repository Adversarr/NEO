"""Multi-resolution point cloud non-uniform sampling for multiple meshes."""

import argparse
import random
from pathlib import Path
from time import perf_counter

import numpy as np

from g2pt.utils.mesh_feats import (
    mesh_laplacian,
    point_cloud_laplacian,
    sample_points_non_uniformly,
    sample_points_uniformly,
    solve_gev,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Sample points non-uniformly at multiple resolutions for multiple meshes.")
    parser.add_argument("--mesh_dir", type=str, required=True, help="Directory containing mesh files.")
    parser.add_argument("--out_dir", type=str, required=True, help="Base directory to save sampled points.")
    parser.add_argument("--n_points", type=int, nargs="+", default=[2048, 8192, 32768], help="List of resolutions to sample.")
    parser.add_argument("--k", type=int, default=128, help="Number of eigenfunctions to compute (should match model's expected k).")
    parser.add_argument("--uniform", action="store_true", help="Use uniform sampling instead (overrides default non-uniform behavior).")
    parser.add_argument("--non_uniform_sigma", type=float, default=None, help="Scale of the dense region for non-uniform sampling.")
    return parser.parse_args()


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    if path.suffix in [".obj", ".ply", ".stl", ".off"]:
        mesh = trimesh.load(path)
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)
    elif path.suffix in [".msh"]:
        import meshio

        mesh_data = meshio.read(path)

        if "triangle" in mesh_data.cells_dict:
            points = mesh_data.points
            faces = mesh_data.cells_dict["triangle"]
        elif "tetra" in mesh_data.cells_dict:
            tetra = mesh_data.cells_dict["tetra"]
            points = mesh_data.points

            faces_local = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
            all_faces = tetra[:, faces_local].reshape(-1, 3)

            sorted_faces = np.sort(all_faces, axis=1)
            _, inverse, counts = np.unique(sorted_faces, return_inverse=True, return_counts=True, axis=0)
            boundary_mask = counts[inverse] == 1
            faces = all_faces[boundary_mask]
        else:
            raise ValueError("Unsupported mesh: no triangle or tetra cells")

        mesh = trimesh.Trimesh(vertices=points, faces=faces, process=True)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    return vertices, faces


def _interpolate_evecs_to_points(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_index: np.ndarray,
    points: np.ndarray,
    evecs: np.ndarray,
) -> np.ndarray:
    tri_vid = faces[face_index]
    tri_verts = vertices[tri_vid]
    v0 = tri_verts[:, 0]
    v1 = tri_verts[:, 1]
    v2 = tri_verts[:, 2]

    area_total = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    area0 = np.linalg.norm(np.cross(v1 - points, v2 - points), axis=1)
    area1 = np.linalg.norm(np.cross(v2 - points, v0 - points), axis=1)
    area2 = np.linalg.norm(np.cross(v0 - points, v1 - points), axis=1)

    eps = 1e-12
    w0 = area0 / (area_total + eps)
    w1 = area1 / (area_total + eps)
    w2 = area2 / (area_total + eps)
    weights = np.stack([w0, w1, w2], axis=1)
    weights = weights / (np.sum(weights, axis=1, keepdims=True) + eps)

    tri_evecs = evecs[tri_vid]
    evecs_on_points = np.sum(tri_evecs * weights[:, :, None], axis=1)

    _, M_pc = point_cloud_laplacian(points)
    diag_M = M_pc.diagonal()
    norm = np.sum(evecs_on_points * evecs_on_points * diag_M[:, None], axis=1, keepdims=True)
    evecs_on_points = evecs_on_points / np.sqrt(norm)
    return evecs_on_points


def main():
    args = parse_args()
    mesh_dir = Path(args.mesh_dir)
    out_dir = Path(args.out_dir)
    
    # Supported mesh extensions
    extensions = [".obj", ".ply", ".stl", ".off", ".msh"]
    mesh_files = [f for f in mesh_dir.iterdir() if f.suffix.lower() in extensions]
    
    if not mesh_files:
        print(f"No mesh files found in {mesh_dir}")
        return

    k = int(args.k)
    for mesh_file in mesh_files:
        mesh_name = mesh_file.stem
        print(f"Processing mesh: {mesh_name}")

        vertices, faces = load_mesh(mesh_file)

        evals = None
        evecs = None
        if k > 0:
            lap_start = perf_counter()
            L, M = mesh_laplacian(vertices, faces)
            lap_end = perf_counter()

            gev_start = perf_counter()
            evals, evecs = solve_gev(L, M, k=k)
            gev_end = perf_counter()

            print(f"⌛️ Mesh Laplacian time: {lap_end - lap_start:.4f} seconds")
            print(f"⌛️ Mesh GEV time: {gev_end - gev_start:.4f} seconds")

        mesh_seed = random.randint(0, 2**31 - 1)

        for n in args.n_points:
            # Output path: <out_dir>/<mesh_name>/<n>/sample_points.npy
            target_dir = out_dir / mesh_name / str(n)
            target_dir.mkdir(parents=True, exist_ok=True)
            input_dir = target_dir / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            out_path = target_dir / "sample_points.npy"

            if args.uniform:
                points, face_index = sample_points_uniformly(
                    vertices, faces, number_of_points=int(n), seed=mesh_seed, return_face_index=True
                )
            else:
                points, face_index = sample_points_non_uniformly(
                    vertices,
                    faces,
                    number_of_points=int(n),
                    seed=mesh_seed,
                    return_face_index=True,
                    sigma=args.non_uniform_sigma,
                )

            np.save(out_path, points)

            import trimesh

            mesh_out_path = input_dir / "sampled_points.ply"
            trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(mesh_out_path)

            pc_out_path = target_dir / "sampled_points_points.ply"
            trimesh.PointCloud(vertices=points).export(pc_out_path)

            if k > 0:
                evecs_on_points = _interpolate_evecs_to_points(vertices, faces, face_index, points, evecs)
                np.save(input_dir / "mesh_eval.npy", np.asarray(evals, dtype=np.float32))
                np.save(input_dir / "mesh_evec.npy", np.asarray(evecs_on_points, dtype=np.float32))

            print(f"✅ Saved {len(points)} points to {out_path}")

if __name__ == "__main__":
    main()
