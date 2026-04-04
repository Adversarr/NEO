import os
from pathlib import Path
import numpy as np
import h5py
from argparse import ArgumentParser

from g2pt.data.common import load_and_process_mesh

def parse_args():
    parser = ArgumentParser(description="Preprocess Human SIG17 segmentation dataset to HDF5 (index labels)")
    parser.add_argument("--root-dir", type=str, required=True, help="Root directory of the Human SIG17 dataset")
    parser.add_argument("--output-dir", type=str, default="/data/processed_human_sig17", help="Output directory for HDF5 files")
    parser.add_argument("--split", type=str, default="all", choices=["train", "test", "val", "all"], help="Which split to process")
    parser.add_argument("--h5-compression", type=str, default="lzf", choices=["none", "gzip", "lzf"], help="HDF5 compression filter for datasets")
    parser.add_argument("--h5-compression-opts", type=int, default=4, help="Compression options (e.g., gzip level); ignored for non-gzip")
    parser.add_argument("--dry-run", action="store_true", help="Enable dry run mode (do not write files)")
    return parser.parse_args()

def collect_pairs(root_dir: Path, split: str):
    """Collect mesh/label file pairs according to SIG17 rules.
    Returns list of tuples: (mesh_path, label_path, obj_id, category)
    obj_id is a string for traceability, typically the basename without extension or dataset-specific id.
    """
    pairs = []
    # Normalize split alias
    split_norm = "test" if split == "val" else split

    if split_norm in ("train", "all"):
        # adobe
        mesh_dir = root_dir / "meshes" / "train" / "adobe"
        label_dir = root_dir / "segs" / "train" / "adobe"
        if mesh_dir.exists() and label_dir.exists():
            for fname in os.listdir(mesh_dir):
                if not fname.lower().endswith((".off", ".obj")):
                    continue
                mesh_path = mesh_dir / fname
                label_path = label_dir / (Path(fname).stem + ".txt")
                if label_path.exists():
                    obj_id = Path(fname).stem
                    pairs.append((mesh_path, label_path, obj_id, "adobe"))

        # faust
        mesh_dir = root_dir / "meshes" / "train" / "faust"
        label_path = root_dir / "segs" / "train" / "faust" / "faust_corrected.txt"
        if mesh_dir.exists() and label_path.exists():
            for fname in os.listdir(mesh_dir):
                if not fname.lower().endswith((".off", ".obj")):
                    continue
                mesh_path = mesh_dir / fname
                obj_id = Path(fname).stem
                pairs.append((mesh_path, label_path, obj_id, "faust"))

        # MIT_animation poses
        pose_names = ['bouncing','handstand','march1','squat1', 'crane','jumping', 'march2', 'squat2']
        for pose in pose_names:
            mesh_dir = root_dir / "meshes" / "train" / "MIT_animation" / f"meshes_{pose}" / "meshes"
            label_path = root_dir / "segs" / "train" / "mit" / f"mit_{pose}_corrected.txt"
            if mesh_dir.exists() and label_path.exists():
                for fname in os.listdir(mesh_dir):
                    if not fname.lower().endswith((".off", ".obj")):
                        continue
                    mesh_path = mesh_dir / fname
                    obj_id = Path(fname).stem
                    pairs.append((mesh_path, label_path, obj_id, f"mit/{pose}"))

        # scape
        mesh_dir = root_dir / "meshes" / "train" / "scape"
        label_path = root_dir / "segs" / "train" / "scape" / "scape_corrected.txt"
        if mesh_dir.exists() and label_path.exists():
            for fname in os.listdir(mesh_dir):
                if not fname.lower().endswith((".off", ".obj")):
                    continue
                mesh_path = mesh_dir / fname
                obj_id = Path(fname).stem
                pairs.append((mesh_path, label_path, obj_id, "scape"))

    if split_norm in ("test", "all"):
        # shrec
        mesh_dir = root_dir / "meshes" / "test" / "shrec"
        label_dir = root_dir / "segs" / "test" / "shrec"
        if mesh_dir.exists() and label_dir.exists():
            for i_shrec in range(1, 21):
                if i_shrec in (16, 18):
                    continue
                if i_shrec == 12:
                    mesh_fname = "12_fix_orientation.off"
                else:
                    mesh_fname = f"{i_shrec}.off"
                label_fname = f"shrec_{i_shrec}_full.txt"
                mesh_path = mesh_dir / mesh_fname
                label_path = label_dir / label_fname
                if mesh_path.exists() and label_path.exists():
                    obj_id = str(i_shrec)
                    pairs.append((mesh_path, label_path, obj_id, "shrec"))

    return pairs

def per_face_to_vertex_index(faces: np.ndarray, face_labels: np.ndarray, n_verts: int, n_class: int = 8) -> np.ndarray:
    """Convert per-face label indices to per-vertex label indices by incident-face voting."""
    # Ensure types
    faces = faces.astype(np.int32)
    face_labels = face_labels.astype(np.int32)
    # Build adjacency: list of classes per vertex
    vert_votes = [[] for _ in range(n_verts)]
    for f_idx, tri in enumerate(faces):
        cls = int(face_labels[f_idx])
        if cls < 0 or cls >= n_class:
            cls = 0
        v0, v1, v2 = int(tri[0]), int(tri[1]), int(tri[2])
        if 0 <= v0 < n_verts: vert_votes[v0].append(cls)
        if 0 <= v1 < n_verts: vert_votes[v1].append(cls)
        if 0 <= v2 < n_verts: vert_votes[v2].append(cls)
    # Vote per vertex
    vert_labels = np.zeros((n_verts,), dtype=np.int32)
    for i in range(n_verts):
        votes = vert_votes[i]
        if not votes:
            vert_labels[i] = 0
        else:
            # count frequencies
            counts = np.bincount(np.array(votes, dtype=np.int32), minlength=n_class)
            # choose smallest index among max
            max_count = counts.max()
            choices = np.where(counts == max_count)[0]
            vert_labels[i] = int(choices.min())
    return vert_labels

def write_mesh_h5(mesh_file: Path, vertices_list: list[np.ndarray], faces_list: list[np.ndarray], obj_paths: list[str], comp: str | None, comp_opts: int | None):
    vlentype_vertices = h5py.special_dtype(vlen=np.dtype("float32"))
    vlentype_faces = h5py.special_dtype(vlen=np.dtype("int32"))
    with h5py.File(str(mesh_file), "w") as hf:
        # obj_paths as bytes
        hf.create_dataset("obj_paths", data=np.array(obj_paths, dtype="S"))
        vert_ds = hf.create_dataset("verts", shape=(len(vertices_list),), dtype=vlentype_vertices, compression=comp, compression_opts=comp_opts)
        face_ds = hf.create_dataset("faces", shape=(len(faces_list),), dtype=vlentype_faces, compression=comp, compression_opts=comp_opts)
        for i, verts in enumerate(vertices_list):
            vert_ds[i] = verts.astype(np.float32).flatten()
        for i, faces in enumerate(faces_list):
            face_ds[i] = faces.astype(np.int32).flatten()
        hf.create_dataset("vert_shapes", data=np.array([v.shape for v in vertices_list], dtype=np.int32))
        hf.create_dataset("face_shapes", data=np.array([f.shape for f in faces_list], dtype=np.int32))

def write_labels_h5(label_file: Path, labels_index_list: list[np.ndarray], face_labels_list: list[np.ndarray], comp: str | None, comp_opts: int | None, n_class: int = 8):
    vlentype_labels = h5py.special_dtype(vlen=np.dtype("int32"))
    with h5py.File(str(label_file), "w") as hf:
        lab_ds = hf.create_dataset("labels", shape=(len(labels_index_list),), dtype=vlentype_labels, compression=comp, compression_opts=comp_opts)
        for i, labels in enumerate(labels_index_list):
            lab_ds[i] = labels.astype(np.int32).flatten()
        hf.create_dataset("label_shapes", data=np.array([[len(lbl)] for lbl in labels_index_list], dtype=np.int32))
        face_lab_ds = hf.create_dataset("face_labels", shape=(len(face_labels_list),), dtype=vlentype_labels, compression=comp, compression_opts=comp_opts)
        for i, f_labels in enumerate(face_labels_list):
            face_lab_ds[i] = f_labels.astype(np.int32).flatten()
        hf.create_dataset("face_label_shapes", data=np.array([[len(lbl)] for lbl in face_labels_list], dtype=np.int32))
        hf.attrs["n_class"] = int(n_class)
        hf.attrs["num_classes"] = int(n_class)
        hf.attrs["index_encoding"] = True

def process_split(pairs: list[tuple[Path, Path, str, str]], output_dir: Path, split_name: str, h5_compression: str, h5_compression_opts: int, dry_run: bool):
    vertices_list: list[np.ndarray] = []
    faces_list: list[np.ndarray] = []
    labels_index_list: list[np.ndarray] = []
    face_labels_list: list[np.ndarray] = []
    obj_paths: list[str] = []

    for mesh_path, label_path, obj_id, category in pairs:
        try:
            ext = mesh_path.suffix.lower()
            file_type = "off" if ext == ".off" else ("obj" if ext == ".obj" else None)
            if file_type is None:
                print(f"Skipping unsupported mesh extension: {mesh_path}")
                continue
            verts, faces, _ = load_and_process_mesh(str(mesh_path), file_type=file_type)
            # normalize to unit cube [-1,1]
            center = np.mean(verts, axis=0)
            verts = verts - center
            max_extent = np.max(np.abs(verts))
            if max_extent > 0:
                verts = verts / max_extent

            face_labels = np.loadtxt(str(label_path)).astype(np.int32) - 1
            if face_labels.ndim != 1:
                face_labels = face_labels.reshape(-1)
            # guard range
            face_labels = np.clip(face_labels, 0, 7)

            # ensure label count aligns with face count
            if faces.shape[0] != face_labels.shape[0]:
                print(f"Skipping label/face mismatch: faces={faces.shape[0]} labels={face_labels.shape[0]} at {mesh_path}")
                continue

            vert_labels = per_face_to_vertex_index(faces, face_labels, n_verts=verts.shape[0], n_class=8)

            vertices_list.append(verts)
            faces_list.append(faces)
            labels_index_list.append(vert_labels)
            face_labels_list.append(face_labels)
            obj_paths.append(f"{category}/{obj_id}")
        except Exception as e:
            print(f"Failed processing {mesh_path}: {e}")
            continue

    if len(vertices_list) == 0:
        print(f"Split {split_name}: No valid objects to write")
        return

    comp = None if h5_compression == "none" else h5_compression
    comp_opts = h5_compression_opts if h5_compression == "gzip" else None

    mesh_out = output_dir / f"human_sig17_{split_name}_mesh.hdf5"
    label_out = output_dir / f"human_sig17_{split_name}_labels.hdf5"

    print(f"Processing split {split_name}")
    print(f"Found {len(pairs)} mesh/label pairs")
    if dry_run:
        print(f"Split {split_name}: Dry run mode")
        print(f"  Would write: {mesh_out}")
        print(f"  Would write: {label_out}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    write_mesh_h5(mesh_out, vertices_list, faces_list, obj_paths, comp, comp_opts)
    write_labels_h5(label_out, labels_index_list, face_labels_list, comp, comp_opts, n_class=8)
    print(f"Split {split_name}: Written {mesh_out}")
    print(f"Split {split_name}: Written {label_out}")

    total_raw_vertices_size = sum(v.nbytes for v in vertices_list)
    total_raw_faces_size = sum(f.nbytes for f in faces_list)
    total_raw_labels_size = sum(l.nbytes for l in labels_index_list)
    total_raw_data_size = total_raw_vertices_size + total_raw_faces_size + total_raw_labels_size

    mesh_file_size = mesh_out.stat().st_size
    label_file_size = label_out.stat().st_size
    total_file_size = mesh_file_size + label_file_size

    print(f"Split {split_name}: Raw data size = {total_raw_data_size/1e6:.2f} MB")
    print(f"Split {split_name}: File size = {total_file_size/1e6:.2f} MB (mesh {mesh_file_size/1e6:.2f} MB, labels {label_file_size/1e6:.2f} MB)")

def main():
    args = parse_args()
    root = Path(args.root_dir)
    out = Path(args.output_dir)
    print(f"Root dir: {root}")
    print(f"Output dir: {out}")
    print(f"Split: {args.split}")
    print(f"Compression: {args.h5_compression} opts={args.h5_compression_opts}")

    if args.split in ("train", "all"):
        train_pairs = collect_pairs(root, "train")
        process_split(train_pairs, out, "train", args.h5_compression, args.h5_compression_opts, args.dry_run)
    if args.split in ("test", "val", "all"):
        test_pairs = collect_pairs(root, "test")
        process_split(test_pairs, out, "test", args.h5_compression, args.h5_compression_opts, args.dry_run)

if __name__ == "__main__":
    main()