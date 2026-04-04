"""
Preprocess the functional mapping.
"""

import os
from pathlib import Path
import numpy as np
import h5py
from argparse import ArgumentParser

from g2pt.data.common import load_and_process_mesh
from tqdm import tqdm
import trimesh
import open3d as o3d
from g2pt.utils.gev import solve_gev_ground_truth
from g2pt.utils.mesh_feats import hks_autoscale, mesh_laplacian

def parse_args():
    parser = ArgumentParser(description="Preprocess functional mapping dataset to HDF5 (index labels)")
    parser.add_argument("--root-dir", type=str, required=True, help="Root directory of the functional mapping dataset")
    parser.add_argument("--output-dir", type=str, default="/data/processed_functional_mapping", help="Output directory for HDF5 files")
    parser.add_argument("--k", type=int, default=128, help="Number of eigenvalues to compute")
    parser.add_argument("--hks-count", type=int, default=16, help="Number of HKS features to compute")
    parser.add_argument('--ntest', type=int, default=20, help="Number of test samples in dataset.")
    parser.add_argument("--h5-compression", type=str, default="lzf", choices=["none", "gzip", "lzf"], help="HDF5 compression filter for datasets")
    parser.add_argument("--h5-compression-opts", type=int, default=4, help="Compression options (e.g., gzip level); ignored for non-gzip")
    parser.add_argument("--dry-run", action="store_true", help="Enable dry run mode (do not write files)")
    return parser.parse_args()

def process_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    k: int = 128,
    hks_count: int = 16
) -> dict:
    """Compute spectral features for a mesh without saving L/M."""
    L, M = mesh_laplacian(vertices, faces)
    total_mass = M.diagonal().sum() / vertices.shape[0]
    M = M / (total_mass + 1e-8) # Ensure mass=1

    evals, evecs = solve_gev_ground_truth(L, M, k)
    hks = hks_autoscale(evecs, evals, hks_count)
    return {
        "mass": M.diagonal().astype(np.float32),
        "evals": evals.astype(np.float32),
        "evecs": evecs.astype(np.float32),
        "hks": hks.astype(np.float32),
    }

def main():
    args = parse_args()
    root = Path(args.root_dir)
    out = Path(args.output_dir)
    print(f"Root dir: {root}")
    print(f"Output dir: {out}")
    print(f"Compression: {args.h5_compression} opts={args.h5_compression_opts}")

    mesh_dir = root / "off_2"
    vts_dir = root / "corres"

    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    corres: list[np.ndarray] = []
    obj_paths: list[str] = []
    for mesh_path in mesh_dir.iterdir():
        if mesh_path.suffix == ".off":
            # mesh = trimesh.load(mesh_path)
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            v, f = np.array(mesh.vertices), np.array(mesh.triangles)
            # Rescale vertices
            v_mean = v.mean(axis=0)
            v = v - v_mean.reshape(1, 3)
            v_scale = np.max(np.abs(v))
            v = v / (v_scale + 1e-6)

            vertices.append(v)
            faces.append(f)
            vts = np.loadtxt(vts_dir / f"{mesh_path.stem}.vts") - 1
            corres.append(vts)
            obj_paths.append(str(mesh_path.relative_to(root)))
            # assert vts.shape[0] == v.shape[0], f"Correspondence shape {vts.shape} != {v.shape}"
            assert np.max(vts) < v.shape[0], f"Correspondence index {np.max(vts)} out of range {v.shape[0]}"
        else:
            print(f"⚠️ Skipping {mesh_path}")

    print(f"💾 Loaded {len(vertices)} meshes")

    # Compute evals/evecs/hks for each mesh
    evals_list: list[np.ndarray] = []
    evecs_list: list[np.ndarray] = []
    hks_list: list[np.ndarray] = []
    mass_list: list[np.ndarray] = []
    for v, f in tqdm(zip(vertices, faces), total=len(vertices)):
        feat = process_mesh(v, f, args.k, args.hks_count)
        evals_list.append(feat["evals"])
        evecs_list.append(feat["evecs"])
        hks_list.append(feat["hks"])
        mass_list.append(feat["mass"])

    # Split train/test by ntest at preprocess stage
    total = len(vertices)
    ntest = max(0, min(args.ntest, total))
    ntrain = total - ntest
    train_idx = list(range(0, ntrain))
    test_idx = list(range(ntrain, total))

    comp = None if args.h5_compression == "none" else args.h5_compression
    comp_opts = args.h5_compression_opts if args.h5_compression == "gzip" else None

    out.mkdir(parents=True, exist_ok=True)

    def write_mesh_h5(mesh_file: Path, idxs: list[int]):
        v_float = h5py.special_dtype(vlen=np.dtype("float32"))
        v_int = h5py.special_dtype(vlen=np.dtype("int32"))
        with h5py.File(str(mesh_file), "w") as hf:
            hf.create_dataset("obj_paths", data=np.array([obj_paths[i] for i in idxs], dtype="S"))
            vds = hf.create_dataset("verts", shape=(len(idxs),), dtype=v_float, compression=comp, compression_opts=comp_opts)
            fds = hf.create_dataset("faces", shape=(len(idxs),), dtype=v_int, compression=comp, compression_opts=comp_opts)
            vert_shapes = []
            face_shapes = []
            for j, i in enumerate(idxs):
                v = vertices[i].astype(np.float32)
                f = faces[i].astype(np.int32)
                vds[j] = v.flatten()
                fds[j] = f.flatten()
                vert_shapes.append(v.shape)
                face_shapes.append(f.shape)
            hf.create_dataset("vert_shapes", data=np.array(vert_shapes, dtype=np.int32))
            hf.create_dataset("face_shapes", data=np.array(face_shapes, dtype=np.int32))

    def write_corr_h5(corr_file: Path, idxs: list[int]):
        v_float = h5py.special_dtype(vlen=np.dtype("float32"))
        v_int = h5py.special_dtype(vlen=np.dtype("int32"))
        with h5py.File(str(corr_file), "w") as hf:
            evals_ds = hf.create_dataset("evals", shape=(len(idxs),), dtype=v_float, compression=comp, compression_opts=comp_opts)
            evecs_ds = hf.create_dataset("evecs", shape=(len(idxs),), dtype=v_float, compression=comp, compression_opts=comp_opts)
            hks_ds = hf.create_dataset("hks", shape=(len(idxs),), dtype=v_float, compression=comp, compression_opts=comp_opts)
            mass_ds = hf.create_dataset("mass", shape=(len(idxs),), dtype=v_float, compression=comp, compression_opts=comp_opts)
            corres_ds = hf.create_dataset("corres", shape=(len(idxs),), dtype=v_int, compression=comp, compression_opts=comp_opts)
            evec_shapes = []
            hks_shapes = []
            mass_shapes = []
            for j, i in enumerate(idxs):
                evals_ds[j] = evals_list[i].astype(np.float32).flatten()
                evecs_ds[j] = evecs_list[i].astype(np.float32).flatten()
                hks_ds[j] = hks_list[i].astype(np.float32).flatten()
                mass_ds[j] = mass_list[i].astype(np.float32).flatten()
                # TODO: `.vts` format may vary across datasets; we assume int indices here.
                corres_ds[j] = np.array(corres[i], dtype=np.int32).flatten()
                evec_shapes.append(evecs_list[i].shape)
                hks_shapes.append(hks_list[i].shape)
                mass_shapes.append(mass_list[i].shape)
            hf.create_dataset("evec_shapes", data=np.array(evec_shapes, dtype=np.int32))
            hf.create_dataset("hks_shapes", data=np.array(hks_shapes, dtype=np.int32))
            hf.create_dataset("mass_shapes", data=np.array(mass_shapes, dtype=np.int32))
            hf.attrs["k"] = int(args.k)
            hf.attrs["hks_count"] = int(args.hks_count)

    train_mesh_out = out / "functional_mapping_train_mesh.hdf5"
    test_mesh_out = out / "functional_mapping_test_mesh.hdf5"
    train_corr_out = out / "functional_mapping_train_corr.hdf5"
    test_corr_out = out / "functional_mapping_test_corr.hdf5"

    if args.dry_run:
        print(f"Dry run: would write {len(train_idx)} train meshes, {len(test_idx)} test meshes")
        print(f"  Mesh train: {train_mesh_out}")
        print(f"  Mesh test:  {test_mesh_out}")
        print(f"  Corr train: {train_corr_out}")
        print(f"  Corr test:  {test_corr_out}")
        return

    write_mesh_h5(train_mesh_out, train_idx)
    write_mesh_h5(test_mesh_out, test_idx)
    write_corr_h5(train_corr_out, train_idx)
    write_corr_h5(test_corr_out, test_idx)
    print(f"Written: {train_mesh_out}")
    print(f"Written: {test_mesh_out}")
    print(f"Written: {train_corr_out}")
    print(f"Written: {test_corr_out}")

if __name__ == "__main__":
    main()