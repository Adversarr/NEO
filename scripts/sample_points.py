from time import perf_counter

from g2pt.utils.mesh_feats import mesh_laplacian, point_cloud_laplacian, sample_points_uniformly, sample_points_non_uniformly, solve_gev
from argparse import ArgumentParser
from pathlib import Path
import numpy as np


def parse_args():
    parser = ArgumentParser(description="Sample points uniformly from a mesh.")
    parser.add_argument("--mesh_path", type=str, required=True, help="Path to the mesh file.")
    parser.add_argument("--n_points", type=int, default=1024, help="Number of points to sample.")
    parser.add_argument("--subdiv", type=int, default=0, help="Number of times to subdivide the mesh.")
    parser.add_argument("--k", type=int, default=96, help="Number of mesh eigenfunctions to compute.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--out", type=str, required=True, help="Path to save the sampled points.")
    parser.add_argument("--non_uniform", action="store_true", help="Use non-uniform sampling (equal probability per face).")
    parser.add_argument("--non_uniform_sigma", type=float, default=None, help="Scale of the dense region for non-uniform sampling.")
    parser.add_argument("--normalize", action="store_true", help="Normalize the points to the unit sphere.")
    return parser.parse_args()


def load_mesh(path: Path, subdiv: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Load any format, return vert, faces as numpy arrays."""
    import trimesh

    if path.suffix in [".obj", ".ply", ".stl", ".off"]:
        mesh = trimesh.load(path)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh)

        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)
    elif path.suffix in [".msh"]:
        import meshio

        # For tetrahedral meshes, extract surface triangles
        mesh_data = meshio.read(path)

        if "triangle" in mesh_data.cells_dict:
            # Already has triangle surface
            vertices = mesh_data.points
            faces = mesh_data.cells_dict["triangle"]
        elif "tetra" in mesh_data.cells_dict:
            # Extract surface from tetrahedral mesh
            tetra = mesh_data.cells_dict["tetra"]
            points = mesh_data.points

            # Generate all faces of tetrahedra
            faces_local = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
            all_faces = tetra[:, faces_local].reshape(-1, 3)

            # Find unique faces (surface faces appear only once)
            sorted_faces = np.sort(all_faces, axis=1)
            _, inverse, counts = np.unique(sorted_faces, return_inverse=True, return_counts=True, axis=0)
            boundary_mask = counts[inverse] == 1
            faces = all_faces[boundary_mask]
        else:
            raise ValueError("Unsupported mesh: no triangle or tetra cells")

        mesh = trimesh.Trimesh(vertices=points, faces=faces, process=True)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    if subdiv > 0:
        mesh = mesh.subdivide(subdiv)

    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    return vertices, faces


def main():
    args = parse_args()
    mesh_path = Path(args.mesh_path)
    out_path = Path(args.out)
    work_dir = out_path.parent
    work_dir.mkdir(parents=True, exist_ok=True)
    input_dir = work_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    vertices, faces = load_mesh(mesh_path, args.subdiv)
    if args.normalize:
        vertices = vertices - np.mean(vertices, axis=0, keepdims=True)
        max_norm = np.max(np.linalg.norm(vertices, axis=1, keepdims=True))
        vertices = vertices / max_norm

    if args.non_uniform:
        points, face_index = sample_points_non_uniformly(
            vertices,
            faces,
            number_of_points=args.n_points,
            seed=args.seed,
            return_face_index=True,
            sigma=args.non_uniform_sigma,
        )
    else:
        points, face_index = sample_points_uniformly(
            vertices,
            faces,
            number_of_points=args.n_points,
            seed=args.seed,
            return_face_index=True,
        )
    np.save(out_path, points)

    import trimesh

    mesh_out_path = input_dir / "sampled_points.ply"
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(mesh_out_path)

    pc_out_path = work_dir / "sampled_points_points.ply"
    trimesh.PointCloud(vertices=points).export(pc_out_path)

    k = int(args.k)
    if k > 0:
        lap_start = perf_counter()
        L, M = mesh_laplacian(vertices, faces)
        lap_end = perf_counter()

        gev_start = perf_counter()
        evals, evecs = solve_gev(L, M, k=k)
        gev_end = perf_counter()

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
        evecs_on_points = np.sum(tri_evecs * weights[:, :, None], axis=1) # (n_points, k)

        # Normalize, use point cloud Laplacian
        L_pc, M_pc = point_cloud_laplacian(points)
        diag_M = M_pc.diagonal()
        norm = np.sum(evecs_on_points * evecs_on_points * diag_M[:, None], axis=1, keepdims=True)
        evecs_on_points = evecs_on_points / np.sqrt(norm)

        np.save(input_dir / "mesh_eval.npy", evals.astype(np.float32))
        np.save(input_dir / "mesh_evec.npy", evecs_on_points.astype(np.float32))

        print(f"⌛️ Mesh Laplacian time: {lap_end - lap_start:.4f} seconds")
        print(f"⌛️ Mesh GEV time: {gev_end - gev_start:.4f} seconds")

    print(f"✅ Saved {len(points)} points to {out_path}")
    print(f"✅ Saved mesh to {mesh_out_path}")
    print(f"✅ Saved point cloud to {pc_out_path}")


if __name__ == "__main__":
    main()
