"""
python scripts/matcher.py \
        --mesh1 ldata/pyFM-off/camel_gallop/camel-gallop-01.off \
        --mesh2 ldata/pyFM-off/camel_gallop/camel-gallop-03.off \
        --feat1 tmp_mesh_infer_4/camel-gallop-01/inferred/net_evec.npy \
        --feat2 tmp_mesh_infer_4/camel-gallop-03/inferred/net_evec.npy \
        --gt_map ldata/pyFM-off/camel_gallop/maps/3_to_1 \
        --out_json camel_comparison.json --out_npz camel_errors.npz \
        --signature=HKS --wks_k=40 --k_process=40 --with_geodesic_error --geodesic_algo graph

python scripts/matcher.py \
        --mesh1 ldata/pyFM-off/camel_gallop/camel-gallop-01.off \
        --mesh2 ldata/pyFM-off/camel_gallop/camel-gallop-05.off \
        --feat1 tmp_mesh_infer_4/camel-gallop-01/inferred/net_evec.npy \
        --feat2 tmp_mesh_infer_4/camel-gallop-05/inferred/net_evec.npy \
        --gt_map ldata/pyFM-off/camel_gallop/maps/1_to_5 \
        --out_json camel_comparison.json --out_npz camel_errors.npz \
        --signature=HKS --wks_k=30 --k_process=64 --with_geodesic_error --geodesic_algo graph

# Good:
python scripts/matcher.py --mesh1 ldata/pyFM-off/cat-00.off --mesh2 ldata/pyFM-off/lion-00.off \
        --feat1 tmp_mesh_infer_5/cat-00/inferred/net_evec.npy --feat2 tmp_mesh_infer_5/lion-00/inferred/net_evec.npy \
        --landmark ldata/pyFM-off/landmarks.txt --gt_map ldata/pyFM-off/lion2cat \
        --out_json catlion_comparison.json --out_npz catlion_errors.npz \
        --signature=WKS --wks_k=30 --with_geodesic_error --geodesic_algo graph --wks_num_E=30 
"""

import igl
import argparse
import numpy as np
import pyFM.eval
import trimesh
from pathlib import Path



from pyFM.mesh import TriMesh
from pyFM.functional import FunctionalMapping
from pyFM.signatures.WKS_functions import mesh_WKS
from pyFM.signatures.HKS_functions import mesh_HKS
from pyFM.spectral.nn_utils import knn_query
from pyFM.refine.zoomout import mesh_zoomout_refine_p2p

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from robust_laplacian import mesh_laplacian
from g2pt.utils.ortho_operations import qr_orthogonalization_numpy
from g2pt.utils.gev import dense_eigsh, solve_gev_ground_truth

import json
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

def load_gt_map(path):
    if path is None:
        return None
    return np.loadtxt(path, dtype=np.int64)


def load_landmarks(path):
    if path is None:
        return None, None
    landmarks = np.loadtxt(path, dtype=np.int64)
    if landmarks.ndim == 1:
        return landmarks, landmarks
    return landmarks[:, 0], landmarks[:, 1]


def build_vertex_edge_graph(vertices, faces):
    n = int(vertices.shape[0])
    faces = np.asarray(faces, dtype=np.int64)

    undirected = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    undirected = np.sort(undirected, axis=1)
    undirected = np.unique(undirected, axis=0)

    w = np.linalg.norm(vertices[undirected[:, 0]] - vertices[undirected[:, 1]], axis=1).astype(np.float64)

    ii = np.concatenate([undirected[:, 0], undirected[:, 1]])
    jj = np.concatenate([undirected[:, 1], undirected[:, 0]])
    ww = np.concatenate([w, w])
    return sp.csr_matrix((ww, (ii, jj)), shape=(n, n))


def compute_geodesic_error(mesh, p2p, gt_map, normalization='area', chunk_size=64, algorithm='graph'):
    if gt_map is None:
        return None

    p2p = np.asarray(p2p, dtype=np.int64).reshape(-1)
    gt_map = np.asarray(gt_map, dtype=np.int64).reshape(-1)
    if p2p.shape != gt_map.shape:
        raise ValueError(f"p2p and gt_map shape mismatch: {p2p.shape} vs {gt_map.shape}")

    unique_sources = np.unique(gt_map)
    errors = np.empty_like(gt_map, dtype=np.float64)

    if algorithm == 'graph':
        graph = build_vertex_edge_graph(mesh.vertices, mesh.faces)
        print(f"Computing graph geodesic distances for {len(unique_sources)} unique GT labels...")
        for start in range(0, len(unique_sources), int(chunk_size)):
            sources = unique_sources[start : start + int(chunk_size)]
            dist = dijkstra(graph, directed=False, indices=sources)

            mask = np.isin(gt_map, sources)
            if not np.any(mask):
                continue
            row_idx = np.searchsorted(sources, gt_map[mask])
            errors[mask] = dist[row_idx, p2p[mask]]
    elif algorithm == 'igl':
        print(f"Computing IGL exact geodesic distances for {len(unique_sources)} unique GT labels (slower)...")
        V = mesh.vertices
        F = mesh.faces
        vt_indices = np.arange(len(V), dtype=np.int64)
        # Process in chunks to avoid excessive memory usage if needed, 
        # but igl.exact_geodesic is generally memory-hungry for multiple sources anyway.
        for start in range(0, len(unique_sources), int(chunk_size)):
            sources = unique_sources[start : start + int(chunk_size)]
            # dist shape: (len(sources), len(V))
            dist = igl.exact_geodesic(
                V.copy().astype(np.float64),
                F.copy().astype(np.float64),
                sources.copy().astype(np.int64),
                np.array([], dtype=np.int64),
                vt_indices.copy().astype(np.int64),
                np.array([], dtype=np.int64),
            )
            
            mask = np.isin(gt_map, sources)
            if not np.any(mask):
                continue
            row_idx = np.searchsorted(sources, gt_map[mask])
            errors[mask] = dist[row_idx, p2p[mask]]
    else:
        raise ValueError(f"Unknown geodesic algorithm: {algorithm}")

    area = float(np.sum(igl.doublearea(mesh.vertices, mesh.faces)) / 2.0)
    if normalization in {"area", "diameter"}:
        norm_factor = float(np.sqrt(max(area, 1e-12)))
    else:
        norm_factor = 1.0

    return errors / norm_factor


def solve_gevp_from_subspace(L, M, features): # TODO: Need checking.
    """
    Solve the generalized eigenvalue problem from the subspace.
    """
    # 1. Ensure orthonormality of the features
    sub = qr_orthogonalization_numpy(features, M.diagonal().reshape(-1, 1))

    L_reduced = sub.T @ (L @ sub)  # [D, D], D is small, dense is ok
    M_reduced = sub.T @ (M @ sub)  # [D, D], not necessary since already normalized.

    num_sub = sub.shape[-1]
    red_eval, red_evec = dense_eigsh(L_reduced, M_reduced)
    reconstructed_evec = (sub @ red_evec)[:, :num_sub]  # [N, k+1]
    return red_eval, reconstructed_evec


def canonicalize_gt_map(gt_map_raw, n_mesh1, n_mesh2):
    gt_map_raw = np.asarray(gt_map_raw, dtype=np.int64).reshape(-1)
    if gt_map_raw.shape[0] == n_mesh2:
        return gt_map_raw, "mesh2_to_mesh1"
    if gt_map_raw.shape[0] == n_mesh1:
        inv = np.full((n_mesh2,), -1, dtype=np.int64)
        for i, j in enumerate(gt_map_raw.tolist()):
            if 0 <= j < n_mesh2 and inv[j] < 0:
                inv[j] = int(i)
        return inv, "mesh1_to_mesh2_inverted"
    raise ValueError(
        f"gt_map length must match mesh1 or mesh2 vertices: {gt_map_raw.shape[0]} vs ({n_mesh1}, {n_mesh2})"
    )


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Mesh matching script using ZoomOut.")
    
    # Mesh paths
    parser.add_argument('--mesh1', type=str, default='../ldata/pyFM-off/cat-00.off', help='Path to the first mesh file')
    parser.add_argument('--mesh2', type=str, default='../ldata/pyFM-off/lion-00.off', help='Path to the second mesh file')

    # Evec Paths
    parser.add_argument('--feat1', type=str, default=None, help='Path to the first mesh features file')
    parser.add_argument('--feat2', type=str, default=None, help='Path to the second mesh features file')
    
    # Mesh processing parameters
    parser.add_argument('--k_process', type=int, default=40, help='Number of eigenfunctions to compute for mesh processing')
    parser.add_argument('--no_center', action='store_false', dest='center', help='Do not center the mesh')
    parser.add_argument('--no_area_normalize', action='store_false', dest='area_normalize', help='Do not normalize mesh area')
    parser.add_argument('--no_intrinsic', action='store_false', dest='intrinsic', help='Do not use intrinsic processing')
    parser.add_argument('--signature', type=str, default='WKS', help='Signature type to use (WKS or HKS)')
    
    # WKS parameters
    parser.add_argument('--wks_num_E', type=int, default=30, help='Number of components for WKS')
    parser.add_argument('--wks_k', type=int, default=30, help='Number of time steps for WKS')
    
    # ZoomOut parameters
    parser.add_argument('--zo_k_init', type=int, default=20, help='Initial dimension for ZoomOut')
    parser.add_argument('--zo_nit', type=int, default=4, help='Number of iterations for ZoomOut')
    parser.add_argument('--zo_step', type=int, default=5, help='Step size for ZoomOut')
    parser.add_argument('--n_jobs', type=int, default=10, help='Number of parallel jobs for ZoomOut')
    parser.add_argument('--no_verbose', action='store_false', dest='verbose', help='Disable verbose output')
    
    # Visualization
    parser.add_argument('--no_plot', action='store_false', dest='plot', help='Disable visualization')
    
    # GT Map and Output
    parser.add_argument('--gt_map', type=str, default=None, help='Path to the ground truth map file')
    parser.add_argument('--with_geodesic_error', action='store_true', help='Compute geodesic error (slow)')
    parser.add_argument('--geodesic_algo', type=str, choices=['graph', 'igl'], default='graph', help='Algorithm for geodesic distance')
    parser.add_argument('--geodesic_chunk_size', type=int, default=64, help='Chunk size for geodesic computation')
    parser.add_argument('--out_json', type=str, default='metrics.json', help='Path to save metrics JSON')
    parser.add_argument('--out_npz', type=str, default='errors.npz', help='Path to save detailed errors NPZ')
    parser.add_argument('--landmarks', type=str, default=None, help='Path to the landmarks file (two columns for mesh1 and mesh2)')
    parser.add_argument('--cmap', type=str, default='viridis', help='Colormap for visualization')
    parser.add_argument('--out_dir', type=Path, default='tmp_out_corr', help='Directory to save output files')



    parser.set_defaults(center=True, area_normalize=True, intrinsic=True, verbose=True, plot=True)
    
    return parser.parse_args()

def _plot_mesh_on_ax(ax, myMesh, cmap, title):
    """
    Helper function to plot a mesh on a given axis.
    """
    vertices = myMesh.vertlist
    faces = myMesh.facelist
    
    if cmap is None:
        cmap = plt.cm.viridis(np.linspace(0, 1, len(vertices)))
    
    face_colors = cmap[faces].mean(axis=1)
    
    mesh = Poly3DCollection(vertices[faces], alpha=0.9)
    mesh.set_facecolor(face_colors)
    mesh.set_edgecolor('k')
    mesh.set_linewidth(0.1)
    
    ax.add_collection3d(mesh)
    
    ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
    ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
    ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

def save_ply_with_colors(path, vertices, faces, colors):
    """
    Save a mesh with vertex colors to a PLY file using trimesh.
    colors: (N, 3) float array in [0, 1]
    """
    colors_u8 = (np.clip(colors, 0, 1) * 255.0).astype(np.uint8)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors_u8)
    mesh.export(path)

def save_combined_plot(mesh1, mesh2, cmap1, cmap2_list, titles, filename="matching_result.png"):
    """
    Plot four meshes in a 2x2 grid and save to file.
    cmap2_list: list of 4 colormaps for mesh2
    """
    fig = plt.figure(figsize=(16, 16))
    
    for i in range(4):
        ax = fig.add_subplot(221 + i, projection='3d')
        _plot_mesh_on_ax(ax, mesh2, cmap2_list[i], titles[i])
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved visualization to {filename}")
    plt.close()

def visu(evec, cmap_name='viridis'):
    """
    Compute color map based on the 1st, 2nd, and 3rd non-trivial eigenvectors.
    The input cmap_name is ignored.
    """
    # Use eigenvectors at indices 1, 2, 3 as RGB channels
    # These are normalized to [0, 1] for visualization
    if evec.shape[1] < 4:
        # Fallback if not enough eigenvectors: use what's available
        colors = np.zeros((evec.shape[0], 3))
        for i in range(min(evec.shape[1], 3)):
            c = evec[:, i]
            colors[:, i] = (c - c.min()) / (c.max() - c.min() + 1e-12)
        return colors

    colors = evec[:, 1:4].copy()
    for i in range(3):
        c = colors[:, i]
        colors[:, i] = (c - c.min()) / (c.max() - c.min() + 1e-12)
    
    return colors

def run_matching_pipeline(mesh1, mesh2, args, landmarks1=None, landmarks2=None):
    """
    Run the full matching pipeline: WKS computation, initial matching, and ZoomOut refinement.
    """
    # Compute WKS signatures
    print(f"Computing {args.signature} signatures...")
    if args.signature == 'WKS':
        wks_descr1 = mesh_WKS(mesh1, num_E=args.wks_num_E, k=args.wks_k, landmarks=landmarks1)
        wks_descr2 = mesh_WKS(mesh2, num_E=args.wks_num_E, k=args.wks_k, landmarks=landmarks2)
    elif args.signature == 'HKS':
        wks_descr1 = mesh_HKS(mesh1, num_T=args.wks_num_E, k=args.wks_k, landmarks=landmarks1)
        wks_descr2 = mesh_HKS(mesh2, num_T=args.wks_num_E, k=args.wks_k, landmarks=landmarks2)
    else:
        raise ValueError(f"Unknown signature type: {args.signature}")

    # Initial correspondences
    print("Computing initial correspondences (WKS)...")
    p2p_21_init = knn_query(wks_descr1, wks_descr2, k=1)

    # Apply ZoomOut
    print("Applying ZoomOut refinement...")
    FM_12_zo, p2p_21_zo = mesh_zoomout_refine_p2p(
        p2p_21=p2p_21_init, mesh1=mesh1, mesh2=mesh2, 
        k_init=args.zo_k_init, nit=args.zo_nit, step=args.zo_step, 
        return_p2p=True, n_jobs=args.n_jobs, verbose=args.verbose
    )
    
    return p2p_21_init, p2p_21_zo

def main():
    args = parse_args()
    
    gt_map = load_gt_map(args.gt_map)
    if gt_map is not None:
        print(f"Loaded ground truth map from {args.gt_map}")

    lm1, lm2 = load_landmarks(args.landmarks)
    if lm1 is not None:
        print(f"Loaded {len(lm1)} landmarks from {args.landmarks}")

    # Load and process meshes
    print(f"Loading Mesh 1: {args.mesh1}")
    print(f"Loading Mesh 2: {args.mesh2}")

    # Prepare two sets of meshes
    # 1. Inferred Eigens
    print("\n--- Processing with Inferred Eigenvectors ---")
    mesh1_inf = TriMesh(args.mesh1, center=True, area_normalize=True).process(k=args.k_process, intrinsic=True)
    mesh2_inf = TriMesh(args.mesh2, center=True, area_normalize=True).process(k=args.k_process, intrinsic=True)
    
    if args.feat1 is not None:
        L1, M1 = mesh_laplacian(mesh1_inf.vertices, mesh1_inf.faces)
        feat1 = np.load(args.feat1)
        evals1, evecs1 = solve_gevp_from_subspace(L1, M1, feat1)
        mesh1_inf.eigenvalues = evals1[:args.k_process]
        mesh1_inf.eigenvectors = evecs1[:, :args.k_process]
    
    if args.feat2 is not None:
        L2, M2 = mesh_laplacian(mesh2_inf.vertices, mesh2_inf.faces)
        feat2 = np.load(args.feat2)
        evals2, evecs2 = solve_gevp_from_subspace(L2, M2, feat2)
        mesh2_inf.eigenvalues = evals2[:args.k_process]
        mesh2_inf.eigenvectors = evecs2[:, :args.k_process]
    
    p2p_inf_init, p2p_inf_zo = run_matching_pipeline(mesh1_inf, mesh2_inf, args, landmarks1=lm1, landmarks2=lm2)
    
    # 2. Ground Truth Eigens
    print("\n--- Processing with Ground Truth Eigenvectors ---")
    mesh1_gt = TriMesh(args.mesh1, center=True, area_normalize=True).process(k=args.k_process, intrinsic=True)
    mesh2_gt = TriMesh(args.mesh2, center=True, area_normalize=True).process(k=args.k_process, intrinsic=True)
    
    p2p_gt_init, p2p_gt_zo = run_matching_pipeline(mesh1_gt, mesh2_gt, args, landmarks1=lm1, landmarks2=lm2)

    # Compute errors if GT map is available
    results = {
        "inputs": {
            "mesh1": args.mesh1,
            "mesh2": args.mesh2,
            "feat1": args.feat1,
            "feat2": args.feat2,
            "gt_map": args.gt_map,
            "landmarks": args.landmarks,
            "k_process": int(args.k_process),
            "signature": args.signature,
        },
        "metrics": {},
    }
    error_data = {
        "p2p_inf_init": p2p_inf_init,
        "p2p_inf_zo": p2p_inf_zo,
        "p2p_gt_init": p2p_gt_init,
        "p2p_gt_zo": p2p_gt_zo,
    }
    
    if gt_map is not None:
        gt_map_raw = np.asarray(gt_map, dtype=np.int64).reshape(-1)
        gt_map_21, gt_mode = canonicalize_gt_map(
            gt_map_raw, n_mesh1=mesh1_inf.vertices.shape[0], n_mesh2=mesh2_inf.vertices.shape[0]
        )
        valid = gt_map_21 >= 0
        valid_frac = float(np.mean(valid.astype(np.float64)))

        acc_inf_init = float(np.mean((p2p_inf_init[valid].astype(np.int64) == gt_map_21[valid]).astype(np.float64)))
        acc_inf_zo = float(np.mean((p2p_inf_zo[valid].astype(np.int64) == gt_map_21[valid]).astype(np.float64)))
        acc_gt_init = float(np.mean((p2p_gt_init[valid].astype(np.int64) == gt_map_21[valid]).astype(np.float64)))
        acc_gt_zo = float(np.mean((p2p_gt_zo[valid].astype(np.int64) == gt_map_21[valid]).astype(np.float64)))

        results["metrics"]["vertex_accuracy"] = {
            "inferred": {"init": acc_inf_init, "zoomout": acc_inf_zo},
            "ground_truth": {"init": acc_gt_init, "zoomout": acc_gt_zo},
        }
        results["metrics"]["gt_map"] = {"mode": gt_mode, "valid_fraction": valid_frac}
        error_data["gt_map_raw"] = gt_map_raw
        error_data["gt_map_21"] = gt_map_21

        print("\nResults:")
        print(f"GT map mode = {gt_mode}, valid fraction = {valid_frac:.6f}")
        print(f"Inferred Eigen: Vertex Acc Init = {acc_inf_init:.6f}, ZoomOut = {acc_inf_zo:.6f}")
        print(f"GT Eigen:       Vertex Acc Init = {acc_gt_init:.6f}, ZoomOut = {acc_gt_zo:.6f}")

        if args.with_geodesic_error:
            print("\n--- Computing Geodesic Errors ---")
            err_inf_init = compute_geodesic_error(
                mesh1_inf, p2p_inf_init[valid], gt_map_21[valid], 
                chunk_size=args.geodesic_chunk_size, algorithm=args.geodesic_algo
            )
            err_inf_zo = compute_geodesic_error(
                mesh1_inf, p2p_inf_zo[valid], gt_map_21[valid], 
                chunk_size=args.geodesic_chunk_size, algorithm=args.geodesic_algo
            )
            err_gt_init = compute_geodesic_error(
                mesh1_gt, p2p_gt_init[valid], gt_map_21[valid], 
                chunk_size=args.geodesic_chunk_size, algorithm=args.geodesic_algo
            )
            err_gt_zo = compute_geodesic_error(
                mesh1_gt, p2p_gt_zo[valid], gt_map_21[valid], 
                chunk_size=args.geodesic_chunk_size, algorithm=args.geodesic_algo
            )

            results["metrics"]["geodesic_error_mean"] = {
                "inferred": {"init": float(np.mean(err_inf_init)), "zoomout": float(np.mean(err_inf_zo))},
                "ground_truth": {"init": float(np.mean(err_gt_init)), "zoomout": float(np.mean(err_gt_zo))},
            }

            error_data["err_inf_init"] = err_inf_init
            error_data["err_inf_zo"] = err_inf_zo
            error_data["err_gt_init"] = err_gt_init
            error_data["err_gt_zo"] = err_gt_zo

            print(
                f"Inferred Eigen: Geodesic Mean Init = {results['metrics']['geodesic_error_mean']['inferred']['init']:.6f}, "
                f"ZoomOut = {results['metrics']['geodesic_error_mean']['inferred']['zoomout']:.6f}"
            )
            print(
                f"GT Eigen:       Geodesic Mean Init = {results['metrics']['geodesic_error_mean']['ground_truth']['init']:.6f}, "
                f"ZoomOut = {results['metrics']['geodesic_error_mean']['ground_truth']['zoomout']:.6f}"
            )

    # Output JSON
    with open(args.out_json, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved metrics to {args.out_json}")
    
    # Output NPZ
    np.savez(args.out_npz, **error_data)
    print(f"Saved detailed errors to {args.out_npz}")

    #evaluaing results
    A_geod = mesh1_gt.get_geodesic(verbose=True)
    # Load an approximate ground truth map
    gt_p2p = np.loadtxt('ldata/pyFM-off/lion2cat',dtype=int)

    # acc_base = pyFM.eval.accuracy(p2p_gt_zo, gt_p2p, A_geod, sqrt_area=mesh1_gt.sqrtarea)
    # print(acc_base * 1e3)

    # Visualization
    if args.plot:
        print("Saving visualization...")
        cmap1 = visu(mesh1_inf.eigenvectors, args.cmap)
        
        # 4 matching results on Mesh 2 transferred from Mesh 1
        cmap2_inf_init = cmap1[p2p_inf_init]
        cmap2_inf_zo = cmap1[p2p_inf_zo]
        cmap2_gt_init = cmap1[p2p_gt_init]
        cmap2_gt_zo = cmap1[p2p_gt_zo]
        
        cmaps = [cmap2_inf_init, cmap2_inf_zo, cmap2_gt_init, cmap2_gt_zo]
        titles = ["Inferred Init", "Inferred ZoomOut", "GT Init", "GT ZoomOut"]
        
        save_combined_plot(mesh1_inf, mesh2_inf, cmap1, cmaps, titles, filename="comparison_result.png")

        args.out_dir.mkdir(parents=True, exist_ok=True)
        save_ply_with_colors(args.out_dir / "res_inf_init.ply", mesh2_inf.vertlist, mesh2_inf.facelist, cmap2_inf_init)
        save_ply_with_colors(args.out_dir / "res_inf_zo.ply", mesh2_inf.vertlist, mesh2_inf.facelist, cmap2_inf_zo)
        save_ply_with_colors(args.out_dir / "res_gt_init.ply", mesh2_inf.vertlist, mesh2_inf.facelist, cmap2_gt_init)
        save_ply_with_colors(args.out_dir / "res_gt_zo.ply", mesh2_inf.vertlist, mesh2_inf.facelist, cmap2_gt_zo)
        save_ply_with_colors(args.out_dir / "source.ply", mesh1_inf.vertlist, mesh1_inf.facelist, cmap1)
        print("Saved matching results to PLY files (res_*.ply and source.ply)")

if __name__ == "__main__":
    main()
