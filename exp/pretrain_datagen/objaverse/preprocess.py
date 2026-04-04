import numpy as np
from argparse import ArgumentParser
import trimesh
from trimesh.exchange.gltf import load_glb
from g2pt.data.objaverse_downloader import load_objects
import h5py
from multiprocessing import Pool
from pathlib import Path
from g2pt.utils.gev import balance_stiffness, solve_gev_ground_truth
from g2pt.utils.mesh_feats import point_cloud_laplacian, sample_points_uniformly
from scipy.sparse import linalg as sla

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input-file", type=str, default="outputs/objaverse_plus_plus.txt", help="Path to the UID file.")
    parser.add_argument("--output-dir", type=str, default="outputs/raw_objaverse", help="Path to the output directory.")
    parser.add_argument("--start", type=int, default=0, help="Start index (inclusive).")
    parser.add_argument("--end", type=int, default=-1, help="End index (exclusive, -1 for all).")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for processing and storage.")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of parallel workers.")
    parser.add_argument("--dry-run", action="store_true", help="Enable dry run mode.")

    parser.add_argument("--k", type=int, default=96, help="Number of eigenvectors to compute; 0 to disable.")
    parser.add_argument("--n-samples", type=int, default=1024, help="Number of points per subsample.")
    parser.add_argument("--per-mesh-count", type=int, default=4, help="Number of subsamples per mesh.")

    parser.add_argument("--delete-after-batch", action="store_true", help="Delete downloaded GLBs after batch writes.")

    parser.add_argument("--min-verts", type=int, default=0, help="Minimum vertex count to keep a mesh.")
    parser.add_argument("--max-verts", type=int, default=500000, help="Maximum vertex count to keep a mesh.")
    parser.add_argument("--max-components", type=int, default=1000, help="Maximum connected components allowed.")
    parser.add_argument("--major-component-min-ratio", type=float, default=0.0, help="Minimum ratio of largest component vertices.")
    parser.add_argument("--reject-non-watertight", action="store_true", help="Reject meshes that are not watertight.")

    parser.add_argument("--h5-compression", type=str, default="lzf", choices=["none", "gzip", "lzf"], help="HDF5 compression filter for datasets")
    parser.add_argument("--h5-compression-opts", type=int, default=4, help="Compression options; only used for gzip")
    return parser.parse_args()


def extract_and_clean_mesh(glb_path, args):
    try:
        with open(glb_path, "rb") as f:
            glb_data = load_glb(f)

        geometries = glb_data.get("geometry", {})
        if not geometries:
            return None, None

        total_v_potential = sum(geom_data.get("vertices", []).shape[0] for _, geom_data in geometries.items())
        if total_v_potential > args.max_verts * 1.1:
            print(f"Reject {glb_path}: total_v_potential={total_v_potential}")
            return None, None

        meshes = []
        for _, geom_data in geometries.items():
            verts = np.array(geom_data.get("vertices", []), dtype=np.float64)
            faces = np.array(geom_data.get("faces", []))
            if verts.size == 0 or faces.size == 0:
                continue
            m = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
            try:
                m.update_faces(m.nondegenerate_faces())
                m.update_faces(m.unique_faces())
                m.remove_unreferenced_vertices()
                m.merge_vertices()
            except Exception:
                pass
            if m.faces.size == 0 or m.vertices.size == 0:
                continue
            meshes.append(m)

        if not meshes:
            return None, None

        combined = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
        try:
            combined.update_faces(combined.nondegenerate_faces())
            combined.update_faces(combined.unique_faces())
            combined.remove_unreferenced_vertices()
            combined.merge_vertices()
        except Exception:
            pass

        if combined.faces.size == 0 or combined.vertices.size == 0:
            return None, None

        components = combined.split(only_watertight=False)
        comp_count = len(components)
        total_v = int(combined.vertices.shape[0])
        largest_v = max((c.vertices.shape[0] for c in components), default=total_v)
        major_ratio = (largest_v / total_v) if total_v > 0 else 0.0

        if total_v < args.min_verts or total_v > args.max_verts:
            print(f"Reject {glb_path}: total_v={total_v}")
            return None, None
        if comp_count > args.max_components:
            print(f"Reject {glb_path}: comp_count={comp_count}")
            return None, None
        if major_ratio < args.major_component_min_ratio:
            print(f"Reject {glb_path}: major_ratio={major_ratio}")
            return None, None
        if args.reject_non_watertight and not combined.is_watertight:
            print(f"Reject {glb_path}: not watertight")
            return None, None

        v = np.asarray(combined.vertices, dtype=np.float32)
        f = np.asarray(combined.faces, dtype=np.int32)
    except Exception:
        return None, None
    print(f"Accept {glb_path}: total_v={total_v}, comp_count={comp_count}, major_ratio={major_ratio}")
    return v, f


def process_single(uid, glb_path, args):
    v, f = extract_and_clean_mesh(glb_path, args)
    if v is None or f is None:
        return None, None, None, None, None

    center = np.mean(v, axis=0)
    v = v - center
    max_extent = np.max(np.abs(v))
    if max_extent > 0:
        v = v / max_extent

    if not np.all(np.isfinite(v)):
        return None, None, None, None, None
    if not np.all(np.isfinite(f)) or not np.all(f >= 0) or np.max(f) >= len(v):
        return None, None, None, None, None

    if args.k == 0:
        return v, f, None, None, None

    try:
        samples_list = []
        evecs_list = []
        mass_list = []
        for _ in range(args.per_mesh_count):
            pts = sample_points_uniformly(v, f, args.n_samples)
            L, M = point_cloud_laplacian(pts)
            eps = 1e-8
            _, evecs = solve_gev_ground_truth(L, M, args.k, dense_threshold=256)
            mass = M.diagonal().reshape(-1, 1)
            norm = np.sqrt(np.sum(evecs * (evecs * mass), axis=0))
            evecs = evecs / norm.reshape(1, -1)
            samples_list.append(pts.astype(np.float32))
            evecs_list.append(evecs.astype(np.float32))
            mass_list.append(M.diagonal().astype(np.float32))
        samples_arr = np.stack(samples_list, axis=0)
        evecs_arr = np.stack(evecs_list, axis=0)
        mass_arr = np.stack(mass_list, axis=0)
        return v, f, samples_arr, evecs_arr, mass_arr
    except Exception:
        return v, f, None, None, None


def process_batch(uids_batch, batch_idx, output_dir, args):
    print(f"Processing batch {batch_idx} with {len(uids_batch)} UIDs using {args.num_workers} workers")
    try:
        desc = load_objects(uids_batch, download_processes=args.num_workers)
    except Exception as e:
        print(f"Batch {batch_idx}: load_objects failed with error: {e}")
        desc = {}
    glb_paths = list(desc.values())
    uids_list = list(desc.keys())
    print(f"Batch {batch_idx}: Request to download {len(uids_batch)} UIDs, success to download {len(glb_paths)} UIDs")

    if args.num_workers > 1:
        try:
            with Pool(processes=args.num_workers) as pool:
                results = pool.starmap(process_single, [(uid, path, args) for uid, path in zip(uids_list, glb_paths)])
        except Exception as e:
            print(f"Batch {batch_idx}: parallel processing failed with error: {e}; falling back to sequential")
            results = [process_single(uid, path, args) for uid, path in zip(uids_list, glb_paths)]
    else:
        results = [process_single(uid, path, args) for uid, path in zip(uids_list, glb_paths)]

    valid_uids = []
    vertices_list = []
    faces_list = []
    samples_list = []
    evecs_list = []
    mass_list = []

    for uid, (v, f, samples, evecs, mass) in zip(uids_list, results):
        if v is None or f is None:
            print(f"Warning: {desc[uid]} contains no valid vertices or faces or was filtered")
            continue
        if len(v) == 0 or len(f) == 0:
            print(f"Warning: {desc[uid]} contains empty vertices or faces")
            continue
        valid_uids.append(uid)
        vertices_list.append(v)
        faces_list.append(f)
        samples_list.append(samples)
        evecs_list.append(evecs)
        mass_list.append(mass)

    if not valid_uids:
        print(f"Batch {batch_idx}: No valid objects found")
        return

    vlentype_vertices = h5py.special_dtype(vlen=np.dtype("float32"))
    vlentype_faces = h5py.special_dtype(vlen=np.dtype("int32"))

    mesh_file = f"{output_dir}/ov_{batch_idx:05}_mesh.hdf5"
    if args.k > 0:
        samples_evecs_file = f"{output_dir}/ov_{batch_idx:05}_samples_evecs.hdf5"

    if args.dry_run:
        print(f"Batch {batch_idx}: Dry run mode")
        print(f"  Would write: {mesh_file}")
        if args.k > 0:
            print(f"  Would write: {samples_evecs_file}")
        return

    try:
        with h5py.File(mesh_file, "w") as hf:
            comp = None if args.h5_compression == "none" else args.h5_compression
            comp_opts = args.h5_compression_opts if args.h5_compression == "gzip" else None
            hf.create_dataset("obj_paths", data=np.array(valid_uids, dtype="S"))
            vds = hf.create_dataset("verts", shape=(len(valid_uids),), dtype=vlentype_vertices, compression=comp, compression_opts=comp_opts)
            fds = hf.create_dataset("faces", shape=(len(valid_uids),), dtype=vlentype_faces, compression=comp, compression_opts=comp_opts)
            for i, (v, f) in enumerate(zip(vertices_list, faces_list)):
                vds[i] = v.astype(np.float32).flatten()
                fds[i] = f.astype(np.int32).flatten()
            hf.create_dataset("vert_shapes", data=np.array([v.shape for v in vertices_list], dtype=np.int32))
            hf.create_dataset("face_shapes", data=np.array([f.shape for f in faces_list], dtype=np.int32))
    except Exception as e:
        print(f"Batch {batch_idx}: failed to write mesh file: {mesh_file}; error: {e}")
        return

    print(f"Batch {batch_idx}: Written {mesh_file}")

    if args.k > 0:
        valid_idx = [i for i, s in enumerate(samples_list) if s is not None]
        if len(valid_idx) > 0:
            try:
                with h5py.File(samples_evecs_file, "w") as hf:
                    comp = None if args.h5_compression == "none" else args.h5_compression
                    comp_opts = args.h5_compression_opts if args.h5_compression == "gzip" else None
                    sel_uids = [valid_uids[i] for i in valid_idx]
                    hf.create_dataset("obj_paths", data=np.array(sel_uids, dtype="S"))
                    samples_array = np.array([samples_list[i] for i in valid_idx], dtype=np.float32)
                    evecs_array = np.array([evecs_list[i] for i in valid_idx], dtype=np.float32)
                    mass_array = np.array([mass_list[i] for i in valid_idx], dtype=np.float32)
                    n_points = int(samples_array.shape[2])
                    channels = int(samples_array.shape[3])
                    hf.create_dataset("samples", data=samples_array, compression=comp, compression_opts=comp_opts, chunks=(1, 1, n_points, channels))
                    kdim = int(evecs_array.shape[3])
                    hf.create_dataset("evecs", data=evecs_array, compression=comp, compression_opts=comp_opts, chunks=(1, 1, n_points, min(64, kdim)))
                    hf.create_dataset("mass", data=mass_array, compression=comp, compression_opts=comp_opts, chunks=(1, 1, n_points))
                    hf.attrs["k"] = args.k
                    hf.attrs["per_mesh_count"] = args.per_mesh_count
                    hf.attrs["n_samples"] = args.n_samples
                print(f"Batch {batch_idx}: Written {samples_evecs_file}")
            except Exception as e:
                print(f"Batch {batch_idx}: failed to write samples/evecs file: {samples_evecs_file}; error: {e}")
        else:
            print(f"Batch {batch_idx}: No valid eigen data to write")

    total_raw_vertices_size = sum(v.nbytes for v in vertices_list)
    total_raw_faces_size = sum(f.nbytes for f in faces_list)
    total_raw_data_size = total_raw_vertices_size + total_raw_faces_size

    try:
        mesh_file_size = Path(mesh_file).stat().st_size
    except Exception as e:
        print(f"Batch {batch_idx}: stat mesh file failed: {mesh_file}; error: {e}")
        mesh_file_size = 0
    total_file_size = mesh_file_size
    if args.k > 0:
        valid_idx = [i for i, s in enumerate(samples_list) if s is not None]
        if len(valid_idx) > 0:
            try:
                samples_evecs_file_size = Path(samples_evecs_file).stat().st_size
                total_file_size += samples_evecs_file_size
            except Exception as e:
                print(f"Batch {batch_idx}: stat samples/evecs file failed: {samples_evecs_file}; error: {e}")

    print(f"\n=== Storage Efficiency Analysis for Batch {batch_idx} ===")
    print("Raw data sizes:")
    print(f"  Vertices: {total_raw_vertices_size / (1024 * 1024):.2f} MB")
    print(f"  Faces: {total_raw_faces_size / (1024 * 1024):.2f} MB")
    print(f"  Total raw data: {total_raw_data_size / (1024 * 1024):.2f} MB")
    print()
    print("Actual file sizes:")
    print(f"  {Path(mesh_file).name}: {mesh_file_size / (1024 * 1024):.2f} MB")
    if args.k > 0:
        if len([i for i, s in enumerate(samples_list) if s is not None]) > 0:
            print(f"  {Path(samples_evecs_file).name}: {samples_evecs_file_size / (1024 * 1024):.2f} MB")
    print(f"  Total file size: {total_file_size / (1024 * 1024):.2f} MB")

    print("Inspect the stored files:")
    try:
        with h5py.File(mesh_file, "r") as hf:
            print(f"  Vertices: {hf['verts'].shape}")
            print(f"  Vertices[0]: {hf['verts'][0].shape}")
            print(f"  Faces: {hf['faces'].shape}")
            print(f"  Faces[0]: {hf['faces'][0].shape}")
            print(f"  Number of objects: {len(hf['obj_paths'])}")
            print(f"  Sample UIDs: {[uid.decode('utf-8')[:20] + '...' if len(uid) > 20 else uid.decode('utf-8') for uid in hf['obj_paths'][:3]]}")
    except Exception as e:
        print(f"Batch {batch_idx}: failed to inspect mesh file: {mesh_file}; error: {e}")

    if args.k > 0:
        valid_idx = [i for i, s in enumerate(samples_list) if s is not None]
        if len(valid_idx) > 0:
            try:
                with h5py.File(samples_evecs_file, "r") as hf:
                    print(f"  Samples: {hf['samples'].shape}")
                    print(f"  Eigenvectors: {hf['evecs'].shape}")
                    print(f"  k: {hf.attrs['k']}")
                    print(f"  per_mesh_count: {hf.attrs['per_mesh_count']}")
                    print(f"  n_samples: {hf.attrs['n_samples']}")
            except Exception as e:
                print(f"Batch {batch_idx}: failed to inspect samples/evecs file: {samples_evecs_file}; error: {e}")

    if args.delete_after_batch:
        deleted = 0
        for p in glb_paths:
            try:
                Path(p).unlink(missing_ok=True)
                deleted += 1
            except Exception:
                pass


def main(args):
    import os
    with open(args.input_file, "r") as f:
        uids_all = [line.strip() for line in f if line.strip()]
    end = len(uids_all) if args.end == -1 else min(args.end, len(uids_all))
    uids = uids_all[args.start:end]
    os.makedirs(args.output_dir, exist_ok=True)
    batch_size = args.batch_size
    num_batches = (len(uids) + batch_size - 1) // batch_size
    print(f"Processing {len(uids)} objects from index {args.start} to {end}")
    print(f"Processing in {num_batches} batches of size {batch_size}")
    for batch_idx in range(num_batches):
        s = batch_idx * batch_size
        e = min((batch_idx + 1) * batch_size, len(uids))
        uids_batch = uids[s:e]
        print(f"\n=== Batch {batch_idx}/{num_batches} ===")
        process_batch(uids_batch, batch_idx, args.output_dir, args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
