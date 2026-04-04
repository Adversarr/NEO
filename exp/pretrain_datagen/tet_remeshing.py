import numpy as np
import h5py
from pathlib import Path
from argparse import ArgumentParser
from multiprocessing import Pool
from g2pt.utils.gev import solve_gev_ground_truth
import meshio
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


def _read_tetmesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = meshio.read(str(path))
    if "tetra" not in mesh.cells_dict:
        raise ValueError(f"Unsupported mesh: no tetra cells in {path}")
    verts = np.asarray(mesh.points, dtype=np.float64)
    tets = np.asarray(mesh.cells_dict["tetra"], dtype=np.int64)
    return verts, tets


def _normalize_points(vertices: np.ndarray) -> np.ndarray:
    center = np.mean(vertices, axis=0, dtype=np.float64)
    vertices = vertices - center
    max_extent = float(np.max(np.abs(vertices)))
    if max_extent > 0:
        vertices = vertices / max_extent
    return vertices


def _tet_boundary_faces(tets: np.ndarray) -> np.ndarray:
    faces_local = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
    all_faces = tets[:, faces_local].reshape(-1, 3)
    sorted_faces = np.sort(all_faces, axis=1)
    _, inverse, counts = np.unique(sorted_faces, return_inverse=True, return_counts=True, axis=0)
    boundary_mask = counts[inverse] == 1
    return all_faces[boundary_mask].astype(np.int64)


def _tet_volumes(vertices: np.ndarray, tets: np.ndarray) -> np.ndarray:
    p0 = vertices[tets[:, 0]]
    p1 = vertices[tets[:, 1]]
    p2 = vertices[tets[:, 2]]
    p3 = vertices[tets[:, 3]]
    d1 = p1 - p0
    d2 = p2 - p0
    d3 = p3 - p0
    det = np.einsum("ij,ij->i", np.cross(d1, d2), d3)
    vol = np.abs(det) / 6.0
    return vol.astype(np.float64)


def _assemble_tet_fem_matrices(vertices: np.ndarray, tets: np.ndarray) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    n = int(vertices.shape[0])
    if tets.size == 0:
        raise ValueError("Empty tetrahedra array")

    p0 = vertices[tets[:, 0]]
    p1 = vertices[tets[:, 1]]
    p2 = vertices[tets[:, 2]]
    p3 = vertices[tets[:, 3]]
    d1 = p1 - p0
    d2 = p2 - p0
    d3 = p3 - p0
    D = np.stack([d1, d2, d3], axis=-1)  # (T, 3, 3)

    det = np.linalg.det(D)
    vol = np.abs(det) / 6.0
    valid = vol > 1e-14
    if not np.any(valid):
        raise ValueError("All tetrahedra are degenerate")

    D = D[valid]
    vol = vol[valid]
    tets = tets[valid]

    invD = np.linalg.inv(D)  # (T, 3, 3)
    invDT = np.transpose(invD, (0, 2, 1))

    g1 = invDT[:, :, 0]
    g2 = invDT[:, :, 1]
    g3 = invDT[:, :, 2]
    g0 = -g1 - g2 - g3
    G = np.stack([g0, g1, g2, g3], axis=1)  # (T, 4, 3)

    K_local = vol[:, None, None] * np.einsum("tik,tjk->tij", G, G)  # (T, 4, 4)
    ones44 = np.ones((4, 4), dtype=np.float64)
    M_local = (vol[:, None, None] / 20.0) * (ones44 + np.eye(4, dtype=np.float64)[None, :, :])

    rows = np.repeat(tets, 4, axis=1)
    cols = np.tile(tets, (1, 4))
    K_data = K_local.reshape(-1)
    M_data = M_local.reshape(-1)
    row = rows.reshape(-1)
    col = cols.reshape(-1)

    K = sp.coo_matrix((K_data, (row, col)), shape=(n, n), dtype=np.float64).tocsr()
    M = sp.coo_matrix((M_data, (row, col)), shape=(n, n), dtype=np.float64).tocsr()
    lumped_mass = np.asarray(M.sum(axis=1)).reshape(-1).astype(np.float64)
    lumped_mass = np.maximum(lumped_mass, 1e-18)
    return K, M, lumped_mass


def _mass_orthonormalize_dense(evecs: np.ndarray, mass: sp.csr_matrix | np.ndarray) -> np.ndarray:
    if isinstance(mass, np.ndarray):
        w = mass.reshape(-1, 1).astype(np.float64)
        gram = evecs.T @ (evecs * w)
    else:
        gram = evecs.T @ (mass @ evecs)
    gram = (gram + gram.T) * 0.5
    gram = gram + np.eye(gram.shape[0], dtype=np.float64) * 1e-10
    L = np.linalg.cholesky(gram)
    evecs = np.linalg.solve(L, evecs.T).T
    return evecs


def _sample_points_in_volume(
    vertices: np.ndarray,
    tets: np.ndarray,
    tet_volumes: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vol_sum = float(np.sum(tet_volumes))
    if not np.isfinite(vol_sum) or vol_sum <= 0:
        raise ValueError("Invalid total volume for sampling")

    cdf = np.cumsum(tet_volumes, dtype=np.float64)
    r = rng.random(n_samples, dtype=np.float64) * cdf[-1]
    tet_idx = np.searchsorted(cdf, r, side="right")
    tet_idx = np.clip(tet_idx, 0, len(tets) - 1)
    chosen = tets[tet_idx]

    u = rng.exponential(scale=1.0, size=(n_samples, 4)).astype(np.float64)
    bary = u / np.sum(u, axis=1, keepdims=True)
    pts = np.einsum("ni,nij->nj", bary, vertices[chosen])
    return pts.astype(np.float32), bary.astype(np.float64), chosen.astype(np.int64)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="/data/processed_tetwild10k")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--k", type=int, default=96)
    parser.add_argument("--n-samples", type=int, default=1024)
    parser.add_argument("--per-mesh-count", type=int, default=4)
    parser.add_argument("--h5-compression", type=str, default="lzf", choices=["none", "gzip", "lzf"])
    parser.add_argument("--h5-compression-opts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def process_single(path: Path, args):
    try:
        vertices, tets = _read_tetmesh(path)
        vertices = _normalize_points(vertices)
        faces = _tet_boundary_faces(tets)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None, None, None, None, None

    if args.k == 0:
        return vertices.astype(np.float32), faces.astype(np.int32), None, None, None

    try:
        K, M, _ = _assemble_tet_fem_matrices(vertices, tets)
        # _, evecs_v = solve_gev_ground_truth(K + 1.0e-8 * M, M, k=args.k)
        ev, evecs_v = eigsh(K, k=args.k, M=M, which="LM", sigma=1.0e-8, tol=1.0e-6, maxiter=1024)
        evecs_v = np.asarray(evecs_v[:, : args.k], dtype=np.float64)
        evecs_v = _mass_orthonormalize_dense(evecs_v, M)
        if evecs_v.shape[1] != args.k:
            raise ValueError(f"Expected {args.k} eigenvectors, got {evecs_v.shape[1]}")

        tet_volumes = _tet_volumes(vertices, tets)
        total_volume = float(np.sum(tet_volumes))
        if not np.isfinite(total_volume) or total_volume <= 0:
            raise ValueError("Invalid mesh volume")
        base_mass = total_volume / float(args.n_samples)

        all_samples = []
        all_evecs = []
        all_mass = []
        rng = np.random.default_rng(int(args.seed) ^ (hash(path.name) & 0xFFFFFFFF))
        for i in range(int(args.per_mesh_count)):
            rng_i = np.random.default_rng(int(rng.integers(0, 2**32 - 1)) ^ int(i))
            pts, bary, chosen = _sample_points_in_volume(vertices, tets, tet_volumes, int(args.n_samples), rng_i)
            evecs_p = np.einsum("ni,nij->nj", bary, evecs_v[chosen]).astype(np.float64)
            mass_p = np.full((int(args.n_samples),), base_mass, dtype=np.float64)
            evecs_p = _mass_orthonormalize_dense(evecs_p, mass_p)

            all_samples.append(pts.astype(np.float32))
            all_evecs.append(evecs_p.astype(np.float32))
            all_mass.append(mass_p.astype(np.float32))

        return (
            vertices.astype(np.float32),
            faces.astype(np.int32),
            np.stack(all_samples, axis=0),
            np.stack(all_evecs, axis=0),
            np.stack(all_mass, axis=0),
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Error processing {path}: {e}")
        return None, None, None, None, None


def process_batch(msh_paths: list[Path], output_dir: Path, batch_idx: int, num_workers: int, dry_run: bool, args):
    if num_workers > 1:
        with Pool(num_workers) as pool:
            results = pool.starmap(process_single, [(p, args) for p in msh_paths])
    else:
        results = [process_single(p, args) for p in msh_paths]

    valid_paths = []
    vertices_list = []
    faces_list = []
    samples_list = []
    evecs_list = []
    mass_list = []

    for path, (vertices, faces, samples, evecs, mass) in zip(msh_paths, results):
        if vertices is None or faces is None:
            continue
        if len(vertices) == 0 or len(faces) == 0:
            continue
        if not np.all(np.isfinite(vertices)):
            continue
        if not np.all(np.isfinite(faces)):
            continue
        if np.min(faces) < 0 or np.max(faces) >= len(vertices):
            continue

        valid_paths.append(path)
        vertices_list.append(vertices)
        faces_list.append(faces)
        samples_list.append(samples)
        evecs_list.append(evecs)
        mass_list.append(mass)

    if not valid_paths:
        print(f"Batch {batch_idx}: No valid meshes")
        return

    vlentype_vertices = h5py.special_dtype(vlen=np.dtype("float32"))
    vlentype_faces = h5py.special_dtype(vlen=np.dtype("int32"))

    mesh_file = f"{output_dir}/twv_{batch_idx:05}_mesh.hdf5"
    samples_evecs_file = f"{output_dir}/twv_{batch_idx:05}_samples_evecs.hdf5" if args.k > 0 else None

    if dry_run:
        print(f"Batch {batch_idx}: Dry run")
        print(f"  Would write: {mesh_file}")
        if samples_evecs_file is not None:
            print(f"  Would write: {samples_evecs_file}")
        return

    with h5py.File(mesh_file, "w") as hf:
        comp = None if args.h5_compression == "none" else args.h5_compression
        comp_opts = args.h5_compression_opts if args.h5_compression == "gzip" else None
        obj_paths = [str(p.name).encode() for p in valid_paths]
        hf.create_dataset("obj_paths", data=np.array(obj_paths, dtype="S"))
        vds = hf.create_dataset("verts", shape=(len(valid_paths),), dtype=vlentype_vertices, compression=comp, compression_opts=comp_opts)
        fds = hf.create_dataset("faces", shape=(len(valid_paths),), dtype=vlentype_faces, compression=comp, compression_opts=comp_opts)
        for i, (v, f) in enumerate(zip(vertices_list, faces_list)):
            vds[i] = v.astype(np.float32).flatten()
            fds[i] = f.astype(np.int32).flatten()
        hf.create_dataset("vert_shapes", data=np.array([v.shape for v in vertices_list], dtype=np.int32))
        hf.create_dataset("face_shapes", data=np.array([f.shape for f in faces_list], dtype=np.int32))

    print(f"Batch {batch_idx}: Written {mesh_file}")

    if samples_evecs_file is not None:
        valid_idx = [i for i, s in enumerate(samples_list) if s is not None]
        if len(valid_idx) > 0:
            samples_array = np.array([samples_list[i] for i in valid_idx]).astype(np.float32)
            evecs_array = np.array([evecs_list[i] for i in valid_idx]).astype(np.float32)
            mass_array = np.array([mass_list[i] for i in valid_idx]).astype(np.float32)

            with h5py.File(samples_evecs_file, "w") as hf:
                comp = None if args.h5_compression == "none" else args.h5_compression
                comp_opts = args.h5_compression_opts if args.h5_compression == "gzip" else None
                obj_paths = [str(valid_paths[i].name).encode() for i in valid_idx]
                hf.create_dataset("obj_paths", data=np.array(obj_paths, dtype="S"))
                n_points = int(samples_array.shape[2])
                channels = int(samples_array.shape[3])
                hf.create_dataset(
                    "samples",
                    data=samples_array,
                    compression=comp,
                    compression_opts=comp_opts,
                    chunks=(1, 1, n_points, channels),
                )
                kdim = int(evecs_array.shape[3])
                hf.create_dataset(
                    "evecs",
                    data=evecs_array,
                    compression=comp,
                    compression_opts=comp_opts,
                    chunks=(1, 1, n_points, min(64, kdim)),
                )
                hf.create_dataset(
                    "mass",
                    data=mass_array,
                    compression=comp,
                    compression_opts=comp_opts,
                    chunks=(1, 1, n_points),
                )
                hf.attrs["k"] = int(args.k)
                hf.attrs["per_mesh_count"] = int(args.per_mesh_count)
                hf.attrs["n_samples"] = int(args.n_samples)

            print(f"Batch {batch_idx}: Written {samples_evecs_file}")


def main(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_msh = sorted(list(input_dir.glob("*.msh")))
    all_msh = all_msh[args.start : args.end if args.end != -1 else len(all_msh)]
    print(f"Found {len(all_msh)} meshes in {input_dir}")

    if args.batch_size == -1:
        print(f"Processing entire dataset as single batch with {len(all_msh)} meshes")
        process_batch(all_msh, output_dir, 0, args.num_workers, args.dry_run, args)
    else:
        num_batches = (len(all_msh) + args.batch_size - 1) // args.batch_size
        for batch_idx in range(num_batches):
            s = batch_idx * args.batch_size
            e = min((batch_idx + 1) * args.batch_size, len(all_msh))
            batch = all_msh[s:e]
            print(f"\n=== Batch {batch_idx}/{num_batches} ===")
            process_batch(batch, output_dir, batch_idx, args.num_workers, args.dry_run, args)

    print(f"Processed {len(all_msh)} meshes.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
