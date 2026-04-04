import numpy as np
import h5py
from pathlib import Path
from argparse import ArgumentParser
from multiprocessing import Pool
from g2pt.utils.gev import balance_stiffness, solve_gev_ground_truth
from g2pt.utils.mesh_feats import point_cloud_laplacian, sample_points_uniformly
import meshio
import trimesh


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
    return parser.parse_args()


def tetmesh_to_surface(path: Path) -> trimesh.Trimesh:
    mesh = meshio.read(str(path))

    if "triangle" in mesh.cells_dict:
        return trimesh.Trimesh(vertices=mesh.points, faces=mesh.cells_dict["triangle"], process=True)

    if "tetra" not in mesh.cells_dict:
        raise ValueError("Unsupported mesh: no triangle or tetra cells")

    tetra = mesh.cells_dict["tetra"]
    points = mesh.points
    faces_local = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    all_faces = tetra[:, faces_local].reshape(-1, 3)
    sorted_faces = np.sort(all_faces, axis=1)
    _, inverse, counts = np.unique(sorted_faces, return_inverse=True, return_counts=True, axis=0)
    boundary_mask = counts[inverse] == 1
    boundary_faces = all_faces[boundary_mask]
    return trimesh.Trimesh(vertices=points, faces=boundary_faces, process=True)


def process_single(path: Path, args):
    try:
        tri = tetmesh_to_surface(path)
        vertices = np.asarray(tri.vertices)
        faces = np.asarray(tri.faces)

        center = np.mean(vertices, axis=0)
        vertices = vertices - center
        max_extent = np.max(np.abs(vertices))
        if max_extent > 0:
            vertices = vertices / max_extent
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None, None, None, None, None

    if args.k == 0:
        return vertices, faces, None, None, None

    try:
        all_samples = []
        all_evecs = []
        all_mass = []
        for _ in range(args.per_mesh_count):
            samples = sample_points_uniformly(vertices, faces, args.n_samples)
            L, M = point_cloud_laplacian(samples)
            L, M = balance_stiffness(L, M, delta=1, k=args.k)
            _, evecs = solve_gev_ground_truth(L, M, k=args.k)
            mass = M.diagonal().reshape(-1, 1)
            norm = np.sqrt(np.sum(evecs * (evecs * mass), axis=0))
            evecs = evecs / (norm.reshape(1, -1) + 1e-12)
            all_samples.append(samples)
            all_evecs.append(evecs[:, : args.k])
            all_mass.append(M.diagonal())

        all_samples = np.stack(all_samples, axis=0)
        all_evecs = np.stack(all_evecs, axis=0)
        all_mass = np.stack(all_mass, axis=0)
        return vertices, faces, all_samples, all_evecs, all_mass
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
        if not np.all(np.isfinite(faces)) or not np.all(faces >= 0) or not np.all(faces == faces.astype(int)):
            continue
        max_vertex_idx = len(vertices) - 1
        if np.max(faces) > max_vertex_idx:
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

    mesh_file = f"{output_dir}/tw_{batch_idx:05}_mesh.hdf5"
    samples_evecs_file = f"{output_dir}/tw_{batch_idx:05}_samples_evecs.hdf5" if args.k > 0 else None

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
                hf.attrs["k"] = args.k
                hf.attrs["per_mesh_count"] = args.per_mesh_count
                hf.attrs["n_samples"] = args.n_samples

            print(f"Batch {batch_idx}: Written {samples_evecs_file}")

    total_raw_vertices_size = sum(v.nbytes for v in vertices_list)
    total_raw_faces_size = sum(f.nbytes for f in faces_list)
    total_raw_data_size = total_raw_vertices_size + total_raw_faces_size

    mesh_file_size = Path(mesh_file).stat().st_size
    total_file_size = mesh_file_size

    if samples_evecs_file is not None and len([s for s in samples_list if s is not None]) > 0:
        total_raw_samples_size = sum(s.nbytes for s in samples_list if s is not None)
        total_raw_evecs_size = sum(e.nbytes for e in evecs_list if e is not None)
        total_raw_data_size += total_raw_samples_size + total_raw_evecs_size
        samples_evecs_file_size = Path(samples_evecs_file).stat().st_size
        total_file_size += samples_evecs_file_size

    print(f"\n=== Storage Efficiency Analysis for Batch {batch_idx} ===")
    print("Raw data sizes:")
    print(f"  Vertices: {total_raw_vertices_size / (1024 * 1024):.2f} MB")
    print(f"  Faces: {total_raw_faces_size / (1024 * 1024):.2f} MB")
    if samples_evecs_file is not None and len([s for s in samples_list if s is not None]) > 0:
        print(f"  Samples: {total_raw_samples_size / (1024 * 1024):.2f} MB")
        print(f"  Eigenvectors: {total_raw_evecs_size / (1024 * 1024):.2f} MB")
    print(f"  Total raw data: {total_raw_data_size / (1024 * 1024):.2f} MB")
    print()
    print("Actual file sizes:")
    print(f"  {Path(mesh_file).name}: {mesh_file_size / (1024 * 1024):.2f} MB")
    if samples_evecs_file is not None and len([s for s in samples_list if s is not None]) > 0:
        print(f"  {Path(samples_evecs_file).name}: {samples_evecs_file_size / (1024 * 1024):.2f} MB")
    print(f"  Total file size: {total_file_size / (1024 * 1024):.2f} MB")

    print("Inspect the stored files:")
    with h5py.File(mesh_file, "r") as hf:
        print(f"  Vertices: {hf['verts'].shape}")
        print(f"  Vertices[0]: {hf['verts'][0].shape}")
        print(f"  Faces: {hf['faces'].shape}")
        print(f"  Faces[0]: {hf['faces'][0].shape}")

    if samples_evecs_file is not None and len([s for s in samples_list if s is not None]) > 0:
        with h5py.File(samples_evecs_file, "r") as hf:
            print(f"  Samples: {hf['samples'].shape}")
            print(f"  Eigenvectors: {hf['evecs'].shape}")
            print(f"  k: {hf.attrs['k']}")
            print(f"  per_mesh_count: {hf.attrs['per_mesh_count']}")
            print(f"  n_samples: {hf.attrs['n_samples']}")


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
