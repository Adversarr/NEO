import numpy as np
from pathlib import Path
from argparse import ArgumentParser
import trimesh
import matplotlib
from matplotlib import colors as mpl_colors

# Try to import extra colormaps if available
try:
    import colorcet
except ImportError:
    pass
try:
    import cmcrameri.cm as cmc
except ImportError:
    pass

def _colormap_colors(values: np.ndarray, cmap_name: str, vmin: float, vmax: float, vcenter: float | None = None):
    """Convert values to RGB colors (0-255 uint8)."""
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    
    if vcenter is not None and (vmin < vcenter < vmax):
        norm = mpl_colors.TwoSlopeNorm(vmin=float(vmin), vcenter=float(vcenter), vmax=float(vmax))
        t = norm(vals)
        t = np.where(np.isfinite(t), t, 0.5)
        t = np.clip(t, 0.0, 1.0)
    else:
        denom = float(vmax - vmin)
        if denom <= 0:
            t = np.full_like(vals, 0.5 if vcenter is not None else 0.0, dtype=np.float64)
        else:
            t = (vals - float(vmin)) / denom
        t = np.clip(t, 0.0, 1.0)

    try:
        cmap = matplotlib.colormaps.get_cmap(cmap_name)
    except ValueError:
        for prefix in ["cmc.", "cet_"]:
            try:
                cmap = matplotlib.colormaps.get_cmap(prefix + cmap_name)
                break
            except ValueError:
                continue
        else:
            raise

    rgba = cmap(t)
    rgb_u8 = (rgba[:, :3] * 255.0).round().astype(np.uint8)
    return rgb_u8

def write_ply_with_attributes(path, vertices, faces=None, attributes=None):
    """
    Writes a PLY file with multiple custom attributes in binary format.
    attributes: dict of {name: np.array of shape (N,)}
    """
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
    ]
    
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    
    if attributes:
        for name in attributes.keys():
            header.append(f"property float {name}")
            dtype.append((name, 'f4'))
            
    if faces is not None:
        header.append(f"element face {len(faces)}")
        header.append("property list uchar int vertex_indices")
        
    header.append("end_header")
    
    vertex_data = np.empty(len(vertices), dtype=dtype)
    vertex_data['x'] = vertices[:, 0].astype(np.float32)
    vertex_data['y'] = vertices[:, 1].astype(np.float32)
    vertex_data['z'] = vertices[:, 2].astype(np.float32)
    
    if attributes:
        for name, data in attributes.items():
            vertex_data[name] = data.astype(np.float32)
            
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode())
        f.write(vertex_data.tobytes())
        
        if faces is not None:
            face_data = np.empty(len(faces), dtype=[('count', 'u1'), ('v1', 'i4'), ('v2', 'i4'), ('v3', 'i4')])
            face_data['count'] = 3
            face_data['v1'] = faces[:, 0].astype(np.int32)
            face_data['v2'] = faces[:, 1].astype(np.int32)
            face_data['v3'] = faces[:, 2].astype(np.int32)
            f.write(face_data.tobytes())

def main():
    parser = ArgumentParser(description="Export PLY with colormapped attributes for Blender.")
    parser.add_argument("--input_dir", type=str, required=True, help="Output directory of infer_mesh.py.")
    parser.add_argument("--k", type=str, default="0", help="Indices of eigenvectors to export.")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory for PLY files.")
    parser.add_argument("--eig_cmap", type=str, default="viridis", help="Colormap name.")
    parser.add_argument("--vmin", type=float, default=None, help="Fixed vmin.")
    parser.add_argument("--vmax", type=float, default=None, help="Fixed vmax.")
    parser.add_argument("--symmetric", action="store_true", help="Symmetric scale around 0.")
    args = parser.parse_args()
    
    input_root = Path(args.input_dir)
    out_dir = Path(args.out_dir) if args.out_dir else input_root / "exported_ply"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if "," in args.k:
            indices = [int(i.strip()) for i in args.k.split(",")]
        elif "-" in args.k:
            start, end = map(int, args.k.split("-"))
            indices = list(range(start, end + 1))
        else:
            indices = [int(args.k)]
    except ValueError:
        print(f"Error parsing k: {args.k}")
        return

    case_dirs = []
    if (input_root / "input").exists() and (input_root / "inferred").exists():
        case_dirs = [input_root]
    else:
        case_dirs = sorted([d for d in input_root.iterdir() if d.is_dir() and (d / "input").exists() and (d / "inferred").exists()])

    print(f"Found {len(case_dirs)} case(s). Exporting indices: {indices}")
    
    for case_dir in case_dirs:
        case_name = case_dir.name
        print(f"Processing case: {case_name}")
        
        try:
            points = np.load(case_dir / "input" / "points.npy")
            faces = None
            if (case_dir / "input" / "faces.npy").exists():
                faces = np.load(case_dir / "input" / "faces.npy")
                if faces.ndim == 1: faces = faces.reshape(-1, 3)
            
            gt_evecs = np.load(case_dir / "inferred" / "mesh_evec.npy")
            pred_evecs = np.load(case_dir / "inferred" / "net_evec.npy")
        except Exception as e:
            print(f"  Warning: Load failed for {case_name}: {e}")
            continue
            
        attributes = {}
        for i, idx in enumerate(indices):
            if idx >= gt_evecs.shape[1] or idx >= pred_evecs.shape[1]:
                continue
                
            gt_k = gt_evecs[:, idx]
            pred_k = pred_evecs[:, idx]
            
            # Scale logic
            vmin = args.vmin if args.vmin is not None else float(min(np.min(gt_k), np.min(pred_k)))
            vmax = args.vmax if args.vmax is not None else float(max(np.max(gt_k), np.max(pred_k)))
            if args.symmetric:
                m = max(abs(vmin), abs(vmax))
                vmin, vmax = -m, m

            # Original values
            attributes[f"gt_{idx}"] = gt_k
            attributes[f"pred_{idx}"] = pred_k
            
            # Colormapped RGB as float attributes (0-1)
            gt_rgb = _colormap_colors(gt_k, args.eig_cmap, vmin, vmax, vcenter=0.0).astype(np.float32) / 255.0
            pred_rgb = _colormap_colors(pred_k, args.eig_cmap, vmin, vmax, vcenter=0.0).astype(np.float32) / 255.0
            
            if i == 0:
                attributes["red"] = gt_rgb[:, 0]
                attributes["green"] = gt_rgb[:, 1]
                attributes["blue"] = gt_rgb[:, 2]

            attributes[f"gt_{idx}_r"] = gt_rgb[:, 0]
            attributes[f"gt_{idx}_g"] = gt_rgb[:, 1]
            attributes[f"gt_{idx}_b"] = gt_rgb[:, 2]
            
            attributes[f"pred_{idx}_r"] = pred_rgb[:, 0]
            attributes[f"pred_{idx}_g"] = pred_rgb[:, 1]
            attributes[f"pred_{idx}_b"] = pred_rgb[:, 2]

        write_ply_with_attributes(out_dir / f"{case_name}.ply", points, faces, attributes)
        print(f"  Exported {case_name}.ply with {len(indices)} indices")

if __name__ == "__main__":
    main()
