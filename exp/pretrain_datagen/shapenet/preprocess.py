import numpy as np
import h5py
from pathlib import Path
from argparse import ArgumentParser
from multiprocessing import Pool
from g2pt.utils.gev import balance_stiffness, solve_gev_ground_truth
from g2pt.utils.mesh_feats import point_cloud_laplacian, sample_points_uniformly
from g2pt.data.common import load_and_process_mesh


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True, help="Path to the ShapeNet dataset directory.")
    parser.add_argument(
        "--output-dir", type=str, default="/data/processed_shapenet", help="Path to the output directory."
    )
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker threads for parallel processing.")
    parser.add_argument("--dry-run", action="store_true", help="Enable dry run mode.")
    parser.add_argument(
        "--categories",
        type=str,
        nargs="*",
        help="Specific categories to process (e.g., 02691156 02808440). Process all if not specified.",
    )

    # Eigens
    parser.add_argument("--k", type=int, default=96, help="Number of eigenvectors to compute.")
    parser.add_argument("--n-samples", type=int, default=1024, help="Number of points to draw per subsample.")
    parser.add_argument("--per-mesh-count", type=int, default=4, help="Number of subsamples per input mesh.")
    parser.add_argument("--h5-compression", type=str, default="lzf", choices=["none", "gzip", "lzf"], help="HDF5 compression filter for datasets")
    parser.add_argument("--h5-compression-opts", type=int, default=4, help="Compression options (e.g., gzip level); ignored for non-gzip")
    return parser.parse_args()


def process_single(path, args):
    """
    1. Load and process the mesh.
    2. Return the normalized vertices (positions), faces
    3. If k > 0, compute samples and eigenvectors
    """
    try:
        # Load and process the mesh using the existing utility
        vertices, faces, mesh = load_and_process_mesh(str(path))

        # Normalize vertices to unit cube [-1, 1]
        # First center the mesh
        center = np.mean(vertices, axis=0)
        vertices = vertices - center

        # Then scale to fit in unit cube
        max_extent = np.max(np.abs(vertices))
        if max_extent > 0:
            vertices = vertices / max_extent  # Now in [-1, 1]
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None, None, None, None, None

    if args.k == 0:
        # Skip eigen computation
        return vertices, faces, None, None, None

    try:
        all_samples = []
        all_evecs = []
        all_mass = []
        for t in range(args.per_mesh_count):
            samples = sample_points_uniformly(vertices, faces, args.n_samples)
            L, M = point_cloud_laplacian(samples)
            L, M = balance_stiffness(L, M, delta=1, k=args.k)
            _, evecs = solve_gev_ground_truth(L, M, k=args.k)
            evecs = evecs[:, : args.k]
            # Ensure ortho about M
            mass = M.diagonal().reshape(-1, 1) # (n, 1)
            norm = np.sqrt(np.sum(evecs * (evecs * mass), axis=0)) # (k)
            evecs = evecs / norm.reshape(1, -1) # (n, k)

            all_samples.append(samples)  # (n_samples, 3)
            all_evecs.append(evecs[:, : args.k])  # (n_samples, k)
            all_mass.append(M.diagonal())  # (n_samples, n_samples)

        # stack at batch dimension
        all_samples = np.stack(all_samples, axis=0)  # (n_mesh, n_samples, 3)
        all_evecs = np.stack(all_evecs, axis=0)  # (n_mesh, n_samples, k)
        all_mass = np.stack(all_mass, axis=0)  # (n_mesh, n_samples, n_samples)
        return vertices, faces, all_samples, all_evecs, all_mass

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Error processing {path}: {e}")
        return None, None, None, None, None


def process_category(category_dir, output_dir, num_workers=1, dry_run=False, args=None):
    """Process all objects in a ShapeNet category"""
    if args is None:
        raise ValueError("args must be provided to process_category")

    category_name = category_dir.name
    print(f"Processing category {category_name}")

    # Find all OBJ files in this category
    all_objs = list(category_dir.glob("*/models/model_normalized.obj"))
    print(f"Found {len(all_objs)} objects in category {category_name}")

    if not all_objs:
        print(f"Category {category_name}: No objects found")
        return

    # Process all objects in the category in parallel
    if num_workers > 1:
        with Pool(num_workers) as pool:
            results = pool.starmap(process_single, [(obj_path, args) for obj_path in all_objs])
    else:
        results = [process_single(obj_path, args) for obj_path in all_objs]

    # Collect valid results
    valid_objs = []
    vertices_list = []
    faces_list = []
    samples_list = []
    evecs_list = []
    mass_list = []

    for obj_path, (vertices, faces, samples, evecs, mass) in zip(all_objs, results):
        if vertices is None or faces is None:
            print(f"Warning: {obj_path} contains no valid vertices or faces")
            continue

        if len(vertices) == 0 or len(faces) == 0:
            print(f"Warning: {obj_path} contains empty vertices or faces")
            continue

        # Check for invalid values (NaN, Inf) in vertices
        if not np.all(np.isfinite(vertices)):
            print(f"Warning: {obj_path} contains invalid values (NaN or Inf) in vertices")
            continue

        # Check for invalid values in faces (NaN, Inf, negative, or non-integer values)
        if not np.all(np.isfinite(faces)) or not np.all(faces >= 0) or not np.all(faces == faces.astype(int)):
            print(f"Warning: {obj_path} contains invalid values in faces (NaN, Inf, or negative indices)")
            continue

        # Check if face indices are within valid range
        max_vertex_idx = len(vertices) - 1
        if np.max(faces) > max_vertex_idx:
            print(f"Warning: {obj_path} contains face indices exceeding vertex count")
            continue

        valid_objs.append(obj_path)
        vertices_list.append(vertices)
        faces_list.append(faces)
        samples_list.append(samples)
        evecs_list.append(evecs)
        mass_list.append(mass)

    if not valid_objs:
        print(f"Category {category_name}: No valid objects found")
        return

    print(f"Category {category_name}: Processing {len(valid_objs)} valid objects")

    # Create variable-length data types for HDF5
    vlentype_vertices = h5py.special_dtype(vlen=np.dtype("float32"))
    vlentype_faces = h5py.special_dtype(vlen=np.dtype("int32"))

    # Store relative paths as object identifiers (category_id/model_id)
    obj_paths = [f"{obj_path.parent.parent.name}/{obj_path.parent.parent.parent.name}" for obj_path in valid_objs]

    # Output: write verts and faces to the same hdf5 file
    mesh_file = f"{output_dir}/sn_{category_name}_mesh.hdf5"

    if args.k > 0:
        samples_evecs_file = f"{output_dir}/sn_{category_name}_samples_evecs.hdf5"

    if dry_run:
        print(f"Category {category_name}: Dry run mode")
        print(f"  Would write: {mesh_file}")
        if args.k > 0:
            print(f"  Would write: {samples_evecs_file}")
        return

    # Write vertex and face data to the same HDF5 file
    with h5py.File(mesh_file, "w") as hf:
        comp = None if args.h5_compression == "none" else args.h5_compression
        comp_opts = args.h5_compression_opts if args.h5_compression == "gzip" else None
        # Store object paths as strings
        hf.create_dataset("obj_paths", data=np.array(obj_paths, dtype="S"))

        # Create dataset for variable-length vertex arrays
        vert_dataset = hf.create_dataset(
            "verts", shape=(len(valid_objs),), dtype=vlentype_vertices, compression=comp, compression_opts=comp_opts
        )

        # Store each object's vertices as a variable-length array
        for i, verts in enumerate(vertices_list):
            # Remember to unflatten after loading
            vert_dataset[i] = verts.astype(np.float32).flatten()

        # Store vertex shapes for reconstruction
        vert_shapes = np.array([v.shape for v in vertices_list], dtype=np.int32)
        hf.create_dataset("vert_shapes", data=vert_shapes)

        # Create dataset for variable-length face arrays
        face_dataset = hf.create_dataset(
            "faces", shape=(len(valid_objs),), dtype=vlentype_faces, compression=comp, compression_opts=comp_opts
        )

        # Store each object's faces as a variable-length array
        for i, faces in enumerate(faces_list):
            # Remember to unflatten after loading
            face_dataset[i] = faces.astype(np.int32).flatten()

        # Store face shapes for reconstruction
        face_shapes = np.array([f.shape for f in faces_list], dtype=np.int32)
        hf.create_dataset("face_shapes", data=face_shapes)

    print(f"Category {category_name}: Written {mesh_file}")

    # Write samples and eigenvectors if k > 0
    if args.k > 0:
        # Filter out objects where eigen computation failed
        valid_samples = [s for s in samples_list if s is not None]
        valid_evecs = [e for e in evecs_list if e is not None]

        if len(valid_samples) > 0 and len(valid_evecs) > 0:
            # Filter mass_list to match valid samples and evecs
            valid_mass = [mass_list[i] for i, s in enumerate(samples_list) if s is not None]
            
            # Convert to numpy arrays (skip objects where eigen computation failed)
            samples_array = np.array(valid_samples)  # Shape: (n_valid_objs, n_mesh, n_samples, 3)
            evecs_array = np.array(valid_evecs)  # Shape: (n_valid_objs, n_mesh, n_samples, k)
            mass_array = np.array(valid_mass)  # Shape: (n_valid_objs, n_mesh, n_samples, 1)

            # Write samples and eigenvectors data to the same HDF5 file
            with h5py.File(samples_evecs_file, "w") as hf:
                comp = None if args.h5_compression == "none" else args.h5_compression
                comp_opts = args.h5_compression_opts if args.h5_compression == "gzip" else None
                # Store object paths
                hf.create_dataset("obj_paths", data=np.array(obj_paths[: len(valid_samples)], dtype="S"))
                
                # Store samples data
                n_points = int(samples_array.shape[2])
                channels = int(samples_array.shape[3])
                hf.create_dataset(
                    "samples",
                    data=samples_array.astype(np.float32),
                    compression=comp,
                    compression_opts=comp_opts,
                    chunks=(1, 1, n_points, channels),
                )
                
                # Store eigenvectors data
                kdim = int(evecs_array.shape[3])
                hf.create_dataset(
                    "evecs",
                    data=evecs_array.astype(np.float32),
                    compression=comp,
                    compression_opts=comp_opts,
                    chunks=(1, 1, n_points, min(64, kdim)),
                )
                
                # Store mass matrix data
                hf.create_dataset(
                    "mass",
                    data=mass_array.astype(np.float32),
                    compression=comp,
                    compression_opts=comp_opts,
                    chunks=(1, 1, n_points),
                )
                
                # Store attributes
                hf.attrs["k"] = args.k
                hf.attrs["per_mesh_count"] = args.per_mesh_count
                hf.attrs["n_samples"] = args.n_samples

            print(f"Category {category_name}: Written {samples_evecs_file}")
        else:
            print(f"Category {category_name}: No valid eigen data to write")

    # Calculate storage efficiency
    total_raw_vertices_size = sum(v.nbytes for v in vertices_list)
    total_raw_faces_size = sum(f.nbytes for f in faces_list)
    total_raw_data_size = total_raw_vertices_size + total_raw_faces_size

    if args.k > 0 and len(valid_samples) > 0:
        total_raw_samples_size = sum(s.nbytes for s in valid_samples)
        total_raw_evecs_size = sum(e.nbytes for e in valid_evecs)
        total_raw_data_size += total_raw_samples_size + total_raw_evecs_size

    # Actual file sizes
    mesh_file_size = Path(mesh_file).stat().st_size
    total_file_size = mesh_file_size

    if args.k > 0 and len(valid_samples) > 0:
        samples_evecs_file_size = Path(samples_evecs_file).stat().st_size
        total_file_size += samples_evecs_file_size

    # Display storage efficiency analysis
    print(f"\n=== Storage Efficiency Analysis for Category {category_name} ===")
    print("Raw data sizes:")
    print(f"  Vertices: {total_raw_vertices_size / (1024 * 1024):.2f} MB")
    print(f"  Faces: {total_raw_faces_size / (1024 * 1024):.2f} MB")
    if args.k > 0 and len(valid_samples) > 0:
        print(f"  Samples: {total_raw_samples_size / (1024 * 1024):.2f} MB")
        print(f"  Eigenvectors: {total_raw_evecs_size / (1024 * 1024):.2f} MB")
    print(f"  Total raw data: {total_raw_data_size / (1024 * 1024):.2f} MB")
    print()
    print("Actual file sizes:")
    print(f"  {Path(mesh_file).name}: {mesh_file_size / (1024 * 1024):.2f} MB")
    if args.k > 0 and len(valid_samples) > 0:
        print(f"  {Path(samples_evecs_file).name}: {samples_evecs_file_size / (1024 * 1024):.2f} MB")
    print(f"  Total file size: {total_file_size / (1024 * 1024):.2f} MB")

    # Inspect the stored files.
    print("Inspect the stored files:")
    with h5py.File(mesh_file, "r") as hf:
        print(f"  Vertices: {hf['verts'].shape}")
        print(f"  Vertices[0]: {hf['verts'][0].shape}")
        print(f"  Faces: {hf['faces'].shape}")
        print(f"  Faces[0]: {hf['faces'][0].shape}")

    if args.k > 0 and len(valid_samples) > 0:
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

    # Find all ShapeNet category directories
    if args.categories:
        # Process specific categories
        categories = [input_dir / cat for cat in args.categories if (input_dir / cat).is_dir()]
        print(f"Processing {len(categories)} specified categories")
    else:
        # Process all categories (directories that contain subdirectories with models)
        categories = [d for d in input_dir.iterdir() if d.is_dir() and list(d.glob("*/models/model_normalized.obj"))]
        print(f"Found {len(categories)} categories in {input_dir}")

    # Process each category
    for category_dir in categories:
        print(f"\n=== Processing Category: {category_dir.name} ===")
        process_category(category_dir, output_dir, args.num_workers, args.dry_run, args)

    print(f"Processed {len(categories)} categories.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
