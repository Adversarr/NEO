"""
$ h5dump -H ./Clock-1/val-00.h5
HDF5 "./Clock-1/val-00.h5" {
GROUP "/" {
   DATASET "insseg_mask" {
      DATATYPE  H5T_IEEE_F32LE
      DATASPACE  SIMPLE { ( 50, 10000 ) / ( 50, 10000 ) }
   }
   DATASET "insseg_one_hot" {
      DATATYPE  H5T_ENUM {
         H5T_STD_I8LE;
         "FALSE"            0;
         "TRUE"             1;
      }
      DATASPACE  SIMPLE { ( 50, 10000, 200 ) / ( 50, 10000, 200 ) }
   }
   DATASET "pts" {
      DATATYPE  H5T_IEEE_F32LE
      DATASPACE  SIMPLE { ( 50, 10000, 3 ) / ( 50, 10000, 3 ) }
   }
   DATASET "semseg_mask" {
      DATATYPE  H5T_IEEE_F32LE
      DATASPACE  SIMPLE { ( 50, 10000 ) / ( 50, 10000 ) }
   }
   DATASET "semseg_one_hot" {
      DATATYPE  H5T_ENUM {
         H5T_STD_I8LE;
         "FALSE"            0;
         "TRUE"             1;
      }
      DATASPACE  SIMPLE { ( 50, 10000, 5 ) / ( 50, 10000, 5 ) }
   }
}
}
"""

from argparse import ArgumentParser
from pathlib import Path
import h5py
import numpy as np
from g2pt.data.transforms import normalize_pc
from g2pt.utils.mesh_feats import point_cloud_laplacian
from multiprocessing import Pool, cpu_count
import json


def parse_args():
    """Parse CLI arguments for PartNet point-cloud preprocessing."""
    parser = ArgumentParser(description="Preprocess PartNet dataset to HDF5 (index labels)")
    parser.add_argument("--data-dir", type=str, required=True, help="Cls directory of the PartNet dataset")
    parser.add_argument("--output-dir", type=str, default="/data/processed_partnet", help="Output directory for HDF5 files")
    parser.add_argument("--h5-compression", type=str, default="lzf", choices=["none", "gzip", "lzf"], help="HDF5 compression filter for datasets")
    parser.add_argument("--h5-compression-opts", type=int, default=4, help="Compression options (e.g., gzip level); ignored for non-gzip")
    parser.add_argument("--dry-run", action="store_true", help="Enable dry run mode (do not write files)")
    parser.add_argument("--num-workers", type=int, default=(cpu_count() or 1), help="Parallel workers for mass computation; set to 1 to disable multiprocessing")
    return parser.parse_args()


def _compression_args(kind: str, opts: int):
    """Map user-friendly compression options to h5py parameters."""
    if kind == "none":
        return None, None
    if kind == "gzip":
        return "gzip", opts
    if kind == "lzf":
        return "lzf", None
    return None, None
    
def _mass_from_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = normalize_pc(points, enable_rotate=0.0).astype(np.float32)
    L, M = point_cloud_laplacian(p)
    m = M.diagonal().astype(np.float32)
    return p, m


def _read_partnet_h5(fp: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Read a raw PartNet H5 file and return (points, labels).

    Points shape: (n_mesh, n_points, 3), dtype float32
    Labels shape: (n_mesh, n_points), dtype int64
    """
    with h5py.File(str(fp), "r") as hf:
        if "pts" not in hf:
            raise RuntimeError(f"Missing 'pts' in {fp}")
        pts = np.array(hf["pts"], dtype=np.float32)
        n_cls = 0
        one = np.array(hf["semseg_one_hot"], dtype=np.int8)
        lbl = one.argmax(axis=-1).astype(np.int64)
        n_cls = hf['semseg_one_hot'].shape[-1]
        if pts.shape[:2] != lbl.shape[:2]:
            raise RuntimeError(f"Shape mismatch points {pts.shape} vs labels {lbl.shape} in {fp}")
        return pts, lbl, n_cls


def _compute_mass_batch(pts: np.ndarray, num_workers: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized points and lumped mass for a batch of point clouds."""
    n_mesh = pts.shape[0]
    if num_workers and num_workers > 1:
        with Pool(processes=num_workers) as pool:
            res = pool.map(_mass_from_points, [pts[i] for i in range(n_mesh)])
    else:
        res = [_mass_from_points(pts[i]) for i in range(n_mesh)]
    P = np.stack([r[0] for r in res], axis=0).astype(np.float32)
    M = np.stack([r[1] for r in res], axis=0).astype(np.float32)
    return P, M


def _write_pc_hdf5(
    out_path: Path, points: np.ndarray, labels: np.ndarray, mass: np.ndarray, comp: str | None, comp_opts: int | None,
    n_cls: int
) -> None:
    """Write points/labels/mass to an HDF5 file compatible with PartNetPointCloudDataset."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(out_path), "w") as hf:
        kw = {}
        if comp is not None:
            kw["compression"] = comp
            if comp_opts is not None:
                kw["compression_opts"] = comp_opts
        hf.create_dataset("points", data=points.astype(np.float32), **kw)
        hf.create_dataset("labels", data=labels.astype(np.int64), **kw)
        hf.create_dataset("mass", data=mass.astype(np.float32), **kw)
        hf.attrs["num_classes"] = n_cls


def _discover_inputs(data_dir: Path) -> list[tuple[str, str, Path]]:
    """Discover input files under data_dir.

    Returns a list of (split, category, path). split is 'train' or 'val'.
    """
    inputs: list[tuple[str, str, Path]] = []
    for cat_dir in sorted([p for p in data_dir.iterdir() if p.is_dir()]):
        cat = cat_dir.name
        for split in ("train", "val"):
            for fp in sorted(cat_dir.glob(f"{split}-*.h5")):
                inputs.append((split, cat, fp))
    for fp in sorted(data_dir.glob("*.h5")):
        name = fp.stem
        if name.startswith("train-"):
            split = "train"
        elif name.startswith("val-") or name.startswith("test-"):
            split = "val"
        else:
            split = "train"
        inputs.append((split, data_dir.name, fp))
    if not inputs:
        raise RuntimeError(f"No PartNet H5 found in {data_dir}")
    return inputs


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    comp, comp_opts = _compression_args(args.h5_compression, args.h5_compression_opts)

    inputs = _discover_inputs(data_dir)
    plan: list[dict] = []
    for split, cat, fp in inputs:
        seg = fp.stem.split("-")[-1]
        out_name = f"{split}-{cat}-{seg}_pc.hdf5"
        out_path = out_dir / out_name
        plan.append({"in": str(fp), "out": str(out_path)})

    if args.dry_run:
        print(json.dumps({"planned": plan}, indent=2))
        return

    for split, cat, fp in inputs:
        if fp.stem.startswith("test"): continue
        pts, lbl, n_cls = _read_partnet_h5(fp)
        P, M = _compute_mass_batch(pts, args.num_workers)
        seg = fp.stem.split("-")[-1]
        out_name = f"{split}-{cat}-{seg}_pc.hdf5"
        out_path = out_dir / out_name
        _write_pc_hdf5(out_path, P, lbl, M, comp, comp_opts, n_cls)
        print(json.dumps({"written": str(out_path), "n_mesh": int(P.shape[0]), "n_points": int(P.shape[1]), "n_cls": int(n_cls)}))


if __name__ == "__main__":
    main()
