"""
This experiment classifies meshes from the SHREC2011 dataset in to 30 categories ('ant', 'pliers', 'laptop', etc).
The dataset contains 20 meshes from each category, for a total of 600 inputs.
The variants of each mesh are nonrigid deformed versions of one another.

shrec_16
├── alien
│   ├── test
│   │   ├── T124.obj
│   │   ├── T411.obj
│   │   ├── T511.obj
│   │   └── T547.obj
│   └── train
│       ├── T133.obj
│       ├── T137.obj
...
"""

from argparse import ArgumentParser
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from g2pt.data.common import load_and_process_mesh


def parse_args():
    p = ArgumentParser(description="Preprocess SHREC16 classification dataset to HDF5 (per-mesh class label)")
    p.add_argument("--input-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="/data/processed_shrec16")
    p.add_argument("--split", type=str, default="all", choices=["train", "test", "all"])
    p.add_argument("--h5-compression", type=str, default="lzf", choices=["none", "gzip", "lzf"])
    p.add_argument("--h5-compression-opts", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def collect_split(input_dir: Path, split: str):
    class_dirs = [d for d in sorted((input_dir).iterdir()) if d.is_dir()]
    entries: list[tuple[Path, int, str]] = []
    for class_idx, cdir in enumerate(class_dirs):
        sub = cdir / split
        if not sub.exists():
            continue
        for fname in os.listdir(sub):
            if not fname.lower().endswith((".obj", ".off")):
                continue
            entries.append((sub / fname, class_idx, cdir.name))
    return entries


def write_mesh_h5(
    mesh_file: Path,
    vertices_list: list[np.ndarray],
    faces_list: list[np.ndarray],
    obj_paths: list[str],
    comp: str | None,
    comp_opts: int | None,
):
    v_float = h5py.special_dtype(vlen=np.dtype("float32"))
    v_int = h5py.special_dtype(vlen=np.dtype("int32"))
    with h5py.File(str(mesh_file), "w") as hf:
        hf.create_dataset("obj_paths", data=np.array(obj_paths, dtype="S"))
        vds = hf.create_dataset(
            "verts", shape=(len(vertices_list),), dtype=v_float, compression=comp, compression_opts=comp_opts
        )
        fds = hf.create_dataset(
            "faces", shape=(len(faces_list),), dtype=v_int, compression=comp, compression_opts=comp_opts
        )
        for i, v in enumerate(vertices_list):
            vds[i] = v.astype(np.float32).flatten()
        for i, f in enumerate(faces_list):
            fds[i] = f.astype(np.int32).flatten()
        hf.create_dataset("vert_shapes", data=np.array([v.shape for v in vertices_list], dtype=np.int32))
        hf.create_dataset("face_shapes", data=np.array([f.shape for f in faces_list], dtype=np.int32))


def write_cls_h5(cls_file: Path, labels: list[int], comp: str | None, comp_opts: int | None, n_class: int):
    with h5py.File(str(cls_file), "w") as hf:
        hf.create_dataset(
            "cls_labels", data=np.array(labels, dtype=np.int32), compression=comp, compression_opts=comp_opts
        )
        hf.attrs["n_class"] = int(n_class)
        hf.attrs["num_classes"] = int(n_class)


def process_split(
    input_dir: Path, output_dir: Path, split: str, h5_compression: str, h5_compression_opts: int, dry_run: bool
):
    entries = collect_split(input_dir, split)
    if len(entries) == 0:
        print(f"Split {split}: no entries")
        return
    
    print(f"Processing split {split}")
    print(f"Found {len(entries)} total mesh files")
    
    vertices_list: list[np.ndarray] = []
    faces_list: list[np.ndarray] = []
    cls_labels: list[int] = []
    obj_paths: list[str] = []
    for mesh_path, class_idx, class_name in entries:
        try:
            ext = mesh_path.suffix.lower()
            ftype = "off" if ext == ".off" else ("obj" if ext == ".obj" else None)
            if ftype is None:
                continue
            verts, faces, _ = load_and_process_mesh(str(mesh_path), file_type=ftype)
            center = np.mean(verts, axis=0)
            verts = verts - center
            max_extent = np.max(np.abs(verts))
            if max_extent > 0:
                verts = verts / max_extent
            vertices_list.append(verts)
            faces_list.append(faces)
            cls_labels.append(int(class_idx))
            obj_paths.append(f"{class_name}/{mesh_path.name}")
        except Exception as e:
            print(f"Failed processing {mesh_path}: {e}")
            continue
    if len(vertices_list) == 0:
        print(f"Split {split}: nothing to write")
        return
    
    # Calculate class distribution
    if cls_labels:
        unique_classes, class_counts = np.unique(cls_labels, return_counts=True)
        print(f"Class distribution: {dict(zip(unique_classes, class_counts))}")
    
    comp = None if h5_compression == "none" else h5_compression
    comp_opts = h5_compression_opts if h5_compression == "gzip" else None
    mesh_out = output_dir / f"shrec16_{split}_mesh.hdf5"
    cls_out = output_dir / f"shrec16_{split}_cls.hdf5"
    
    if dry_run:
        print(f"Split {split}: Dry run mode")
        print(f"  Would write: {mesh_out}")
        print(f"  Would write: {cls_out}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    write_mesh_h5(mesh_out, vertices_list, faces_list, obj_paths, comp, comp_opts)
    n_class = int(max(cls_labels) + 1) if cls_labels else 0
    write_cls_h5(cls_out, cls_labels, comp, comp_opts, n_class=n_class)
    print(f"Split {split}: Written {mesh_out}")
    print(f"Split {split}: Written {cls_out}")
    
    # Add detailed size statistics
    total_raw_vertices_size = sum(v.nbytes for v in vertices_list)
    total_raw_faces_size = sum(f.nbytes for f in faces_list)
    total_raw_labels_size = sum(np.array([lbl]).nbytes for lbl in cls_labels)
    total_raw_data_size = total_raw_vertices_size + total_raw_faces_size + total_raw_labels_size
    
    mesh_file_size = mesh_out.stat().st_size
    cls_file_size = cls_out.stat().st_size
    total_file_size = mesh_file_size + cls_file_size
    
    print(f"Split {split}: Raw data size = {total_raw_data_size/1e6:.2f} MB")
    print(f"Split {split}: File size = {total_file_size/1e6:.2f} MB (mesh {mesh_file_size/1e6:.2f} MB, class labels {cls_file_size/1e6:.2f} MB)")


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    out = Path(args.output_dir)
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {out}")
    print(f"Split: {args.split}")
    print(f"Compression: {args.h5_compression} opts={args.h5_compression_opts}")
    
    if args.split in ("train", "all"):
        process_split(input_dir, out, "train", args.h5_compression, args.h5_compression_opts, args.dry_run)
    if args.split in ("test", "all"):
        process_split(input_dir, out, "test", args.h5_compression, args.h5_compression_opts, args.dry_run)


if __name__ == "__main__":
    main()
