"""Precompute normal vector for shapenet"""

import numpy as np
import h5py
import open3d as o3d
from pathlib import Path
from argparse import ArgumentParser
from multiprocessing import Pool
import trimesh
from tqdm import tqdm
from g2pt.data.common import load_and_process_mesh
from g2pt.utils.mesh_feats import sample_points_uniformly


def parse_args():
    parser = ArgumentParser(description="Precompute surface normals and interpolate to point clouds for ShapeNet.")
    parser.add_argument("--input-dir", type=str, required=True, help="Path to the ShapeNet dataset directory.")
    parser.add_argument(
        "--output-dir", type=str, default="/data/processed_shapenet_normals", help="Path to the output directory."
    )
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker threads for parallel processing.")
    parser.add_argument("--dry-run", action="store_true", help="Enable dry run mode.")
    parser.add_argument(
        "--categories",
        type=str,
        nargs="*",
        help="Specific categories to process (e.g., 02691156 02808440). Process all if not specified.",
    )

    # Sampling and interpolation
    parser.add_argument("--n-samples", type=int, default=1024, help="Number of points to sample per mesh.")
    parser.add_argument("--per-mesh-count", type=int, default=1, help="Number of subsamples per input mesh.")
    parser.add_argument(
        "--h5-compression",
        type=str,
        default="lzf",
        choices=["none", "gzip", "lzf"],
        help="HDF5 compression filter for datasets",
    )
    parser.add_argument(
        "--h5-compression-opts", type=int, default=4, help="Compression options (e.g., gzip level); ignored for non-gzip"
    )
    return parser.parse_args()


def _new_o3d_triangle_mesh():
    """Create an Open3D TriangleMesh on CPU if available."""
    try:
        return o3d.cpu.pybind.geometry.TriangleMesh()
    except Exception:
        return o3d.geometry.TriangleMesh()


def interpolate_normals(mesh, points, face_indices):
    """
    Interpolate vertex normals to the sampled points using barycentric coordinates.
    
    Args:
        mesh (trimesh.Trimesh): The mesh object with vertex_normals.
        points (np.ndarray): Sampled points on the surface.
        face_indices (np.ndarray): Indices of faces from which points were sampled.
        
    Returns:
        np.ndarray: Interpolated and normalized normals for each point.
    """
    face_normals = mesh.face_normals[face_indices]

    faces = mesh.faces[face_indices]
    n0 = mesh.vertex_normals[faces[:, 0]]
    n1 = mesh.vertex_normals[faces[:, 1]]
    n2 = mesh.vertex_normals[faces[:, 2]]

    triangles = mesh.triangles[face_indices]
    bary = trimesh.triangles.points_to_barycentric(triangles, points)

    interpolated_normals = (
        n0 * bary[:, 0:1] +
        n1 * bary[:, 1:2] +
        n2 * bary[:, 2:3]
    ).astype(np.float32)

    norm = np.linalg.norm(interpolated_normals, axis=1, keepdims=True).astype(np.float32)
    
    bad_mask = (norm < 1e-6).flatten()
    if np.any(bad_mask):
        interpolated_normals[bad_mask] = face_normals[bad_mask]
        norm[bad_mask] = 1.0

    interpolated_normals = interpolated_normals / (norm + 1e-8)

    return interpolated_normals


def process_single(path, args):
    """
    Process a single mesh: load, fix normals, sample points, and interpolate normals.
    """
    try:
        vertices, faces, _mesh = load_and_process_mesh(str(path))
        vertices = vertices.astype(np.float32)
        faces = faces.astype(np.int32)

        center = np.mean(vertices, axis=0).astype(np.float32)
        vertices = vertices - center
        max_extent = np.max(np.abs(vertices)).astype(np.float32)
        if max_extent > 0:
            vertices = (vertices / max_extent).astype(np.float32)

        mesh_o3d = _new_o3d_triangle_mesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
        mesh_o3d.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        mesh_o3d.remove_duplicated_vertices()
        mesh_o3d.remove_duplicated_triangles()
        
        if hasattr(mesh_o3d, "is_orientable") and mesh_o3d.is_orientable() and hasattr(mesh_o3d, "orient_triangles"):
            mesh_o3d.orient_triangles()
        
        vertices = np.asarray(mesh_o3d.vertices).astype(np.float32)
        faces = np.asarray(mesh_o3d.triangles).astype(np.int32)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        face_centers = mesh.triangles.mean(axis=1)
        dots = (mesh.face_normals * face_centers).sum(axis=1)
        
        valid = np.linalg.norm(face_centers, axis=1) > 1e-3
        if np.any(valid):
            if np.sum(dots[valid] < 0) > np.sum(dots[valid] > 0):
                mesh.invert()
        else:
            extents = mesh.bounding_box.extents
            axis = np.argmin(extents)
            if np.mean(mesh.face_normals[:, axis]) < 0:
                mesh.invert()

        vertices = mesh.vertices.astype(np.float32)
        faces = mesh.faces.astype(np.int32)

        all_samples = []
        all_normals = []

        for t in range(args.per_mesh_count):
            samples, face_indices = sample_points_uniformly(
                vertices, faces, args.n_samples, seed=42 + t, return_face_index=True
            )

            normals = interpolate_normals(mesh, samples, face_indices)
            
            all_samples.append(samples)
            all_normals.append(normals)

        all_samples = np.stack(all_samples, axis=0)
        all_normals = np.stack(all_normals, axis=0)

        return vertices, faces, all_samples, all_normals

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error processing {path}: {e}")
        return None, None, None, None


def process_category(category_dir, output_dir, num_workers=1, dry_run=False, args=None):
    """Process all objects in a ShapeNet category and save to HDF5."""
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

    # Process in parallel
    if num_workers > 1:
        with Pool(num_workers) as pool:
            # Use tqdm to monitor progress of pool.starmap
            results = list(tqdm(
                pool.starmap(process_single, [(obj_path, args) for obj_path in all_objs]),
                total=len(all_objs),
                desc=f"  Processing {category_name}",
                leave=False
            ))
    else:
        results = [process_single(obj_path, args) for obj_path in tqdm(all_objs, desc=f"  Processing {category_name}", leave=False)]

    # Collect valid results
    valid_objs = []
    vertices_list = []
    faces_list = []
    samples_list = []
    normals_list = []

    for obj_path, (vertices, faces, samples, normals) in zip(all_objs, results):
        if vertices is None or faces is None or samples is None:
            continue

        valid_objs.append(obj_path)
        vertices_list.append(vertices)
        faces_list.append(faces)
        samples_list.append(samples)
        normals_list.append(normals)

    if not valid_objs:
        print(f"Category {category_name}: No valid objects found")
        return

    print(f"Category {category_name}: Processing {len(valid_objs)} valid objects")

    # Store relative paths as object identifiers (category_id/model_id)
    obj_paths = [f"{obj_path.parent.parent.name}/{obj_path.parent.parent.parent.name}" for obj_path in valid_objs]

    # Output file path
    output_file = f"{output_dir}/sn_{category_name}_normals.hdf5"

    if dry_run:
        print(f"Category {category_name}: Dry run mode. Would write: {output_file}")
        return

    # Write to HDF5
    with h5py.File(output_file, "w") as hf:
        comp = None if args.h5_compression == "none" else args.h5_compression
        comp_opts = args.h5_compression_opts if args.h5_compression == "gzip" else None
        
        hf.create_dataset("obj_paths", data=np.array(obj_paths, dtype="S"))
        
        # Store samples and normals
        samples_array = np.array(samples_list, dtype=np.float32)
        normals_array = np.array(normals_list, dtype=np.float32)
        
        hf.create_dataset("samples", data=samples_array, compression=comp, compression_opts=comp_opts)
        hf.create_dataset("normals", data=normals_array, compression=comp, compression_opts=comp_opts)
        
        # Optionally store original mesh data if needed, but for now we focus on normals
        # We can use variable length types if we want to store original vertices/faces
        vlentype_vertices = h5py.special_dtype(vlen=np.dtype("float32"))
        vlentype_faces = h5py.special_dtype(vlen=np.dtype("int32"))
        
        vert_dataset = hf.create_dataset(
            "verts", shape=(len(valid_objs),), dtype=vlentype_vertices, compression=comp, compression_opts=comp_opts
        )
        for i, v in enumerate(vertices_list):
            vert_dataset[i] = v.astype(np.float32).flatten()
            
        face_dataset = hf.create_dataset(
            "faces", shape=(len(valid_objs),), dtype=vlentype_faces, compression=comp, compression_opts=comp_opts
        )
        for i, f in enumerate(faces_list):
            face_dataset[i] = f.astype(np.int32).flatten()
            
        hf.create_dataset("vert_shapes", data=np.array([v.shape for v in vertices_list], dtype=np.int32))
        hf.create_dataset("face_shapes", data=np.array([f.shape for f in faces_list], dtype=np.int32))

        # Attributes
        hf.attrs["n_samples"] = args.n_samples
        hf.attrs["per_mesh_count"] = args.per_mesh_count

    print(f"Category {category_name}: Written {output_file}")


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find categories
    if args.categories:
        categories = [input_dir / cat for cat in args.categories if (input_dir / cat).is_dir()]
    else:
        categories = [d for d in input_dir.iterdir() if d.is_dir() and list(d.glob("*/models/model_normalized.obj"))]

    print(f"Found {len(categories)} categories.")

    # Process each category
    pbar = tqdm(categories, desc="Processing categories")
    for category_dir in pbar:
        pbar.set_postfix({"category": category_dir.name})
        process_category(category_dir, output_dir, args.num_workers, args.dry_run, args)


if __name__ == "__main__":
    main()
