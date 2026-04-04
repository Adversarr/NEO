from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def _load_sample(
    samples_h5: Path,
    mesh_index: int,
    subsample_index: int,
    evec_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(samples_h5, "r") as hf:
        points = np.array(hf["samples"][mesh_index, subsample_index], dtype=np.float32)
        mass = np.array(hf["mass"][mesh_index, subsample_index], dtype=np.float32)
        evecs = np.array(hf["evecs"][mesh_index, subsample_index, :, :evec_dim], dtype=np.float32)
    return points, mass, evecs


def _load_surface_mesh(mesh_h5: Path, mesh_index: int) -> tuple[np.ndarray, np.ndarray] | None:
    with h5py.File(mesh_h5, "r") as hf:
        v_shape = tuple(hf["vert_shapes"][mesh_index].tolist())
        f_shape = tuple(hf["face_shapes"][mesh_index].tolist())
        v_flat = np.array(hf["verts"][mesh_index], dtype=np.float32)
        f_flat = np.array(hf["faces"][mesh_index], dtype=np.int32)
    verts = v_flat.reshape(v_shape)
    faces = f_flat.reshape(f_shape)
    if verts.size == 0 or faces.size == 0:
        return None
    return verts, faces


def _axis_equal_3d(ax):
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    xmid = 0.5 * (xlim[0] + xlim[1])
    ymid = 0.5 * (ylim[0] + ylim[1])
    zmid = 0.5 * (zlim[0] + zlim[1])
    r = 0.5 * max(xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0])
    ax.set_xlim3d(xmid - r, xmid + r)
    ax.set_ylim3d(ymid - r, ymid + r)
    ax.set_zlim3d(zmid - r, zmid + r)


def main():
    parser = ArgumentParser(description="Visualize one sample from tet-remeshing HDF5 dataset and save image.")
    parser.add_argument("--samples-h5", type=str, required=True, help="Path to *_samples_evecs.hdf5")
    parser.add_argument("--mesh-h5", type=str, default="", help="Optional path to *_mesh.hdf5 for surface overlay")
    parser.add_argument("--mesh-index", type=int, default=0, help="Mesh index within the file")
    parser.add_argument("--subsample-index", type=int, default=0, help="Subsample index within per_mesh_count")
    parser.add_argument("--evec-dim", type=int, default=8, help="How many eigenvector dims to load")
    parser.add_argument("--color", type=str, default="evec", choices=["evec", "mass", "none"], help="Color mode")
    parser.add_argument("--evec-index", type=int, default=1, help="Which eigenvector component to color")
    parser.add_argument("--point-size", type=float, default=2.0, help="Point size for scatter")
    parser.add_argument("--alpha", type=float, default=1.0, help="Point alpha")
    parser.add_argument("--elev", type=float, default=20.0, help="Camera elevation")
    parser.add_argument("--azim", type=float, default=45.0, help="Camera azimuth")
    parser.add_argument("--dpi", type=int, default=300, help="Save DPI")
    parser.add_argument("--out", type=str, required=True, help="Output image path (e.g., out.png)")
    args = parser.parse_args()

    samples_h5 = Path(args.samples_h5)
    mesh_h5 = Path(args.mesh_h5) if args.mesh_h5 else None
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    points, mass, evecs = _load_sample(
        samples_h5=samples_h5,
        mesh_index=int(args.mesh_index),
        subsample_index=int(args.subsample_index),
        evec_dim=int(args.evec_dim),
    )

    colors = None
    cmap = None
    if args.color == "mass":
        colors = mass.astype(np.float32)
        cmap = "viridis"
    elif args.color == "evec":
        idx = int(args.evec_index)
        if idx < 0 or idx >= evecs.shape[1]:
            raise ValueError(f"evec-index {idx} out of bounds for loaded evec_dim={evecs.shape[1]}")
        colors = evecs[:, idx].astype(np.float32)
        cmap = "coolwarm"

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors,
        cmap=cmap,
        s=float(args.point_size),
        alpha=float(args.alpha),
        linewidths=0,
    )
    ax.view_init(elev=float(args.elev), azim=float(args.azim))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    _axis_equal_3d(ax)

    if colors is not None:
        fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)

    if mesh_h5 is not None and mesh_h5.exists():
        mesh_data = _load_surface_mesh(mesh_h5, int(args.mesh_index))
        if mesh_data is not None:
            verts, faces = mesh_data
            ax.plot_trisurf(
                verts[:, 0],
                verts[:, 1],
                verts[:, 2],
                triangles=faces,
                color=(0.6, 0.6, 0.6, 0.15),
                linewidth=0.2,
                edgecolor=(0.2, 0.2, 0.2, 0.05),
            )

    title = f"mesh={args.mesh_index} subsample={args.subsample_index} N={points.shape[0]}"
    if args.color == "evec":
        title += f" color=evec[{args.evec_index}]"
    elif args.color == "mass":
        title += " color=mass"
    ax.set_title(title)

    plt.savefig(str(out_path), dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
