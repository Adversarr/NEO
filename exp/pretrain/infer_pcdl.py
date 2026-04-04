from time import perf_counter
from pathlib import Path
from argparse import ArgumentParser
import json

import numpy as np
import scipy.sparse as sp
import torch

from g2pt.training.pretrain import PretrainTraining
from g2pt.utils.gev import outer_cosine_similarity, solve_gev_from_subspace_cuda, solve_gev_ground_truth
from g2pt.data.transforms import normalize_pc
from pypcdlaplace import pcdlp_matrix

import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp


def compute_surface_lbo(X, k=20, simple_mass=False):
    """
    Constructs the robust Laplace-Beltrami Operator for a surface point cloud.
    Implements the diffusion maps normalization to handle non-uniform sampling.
    
    Args:
        X: (N, 3) point cloud.
        k: k-NN parameter.
        simple_mass: If True, M=I (not recommended for FEM-like interpretation).
                     If False, M = diag(local_area).
    Returns:
        L (csc_matrix): Stiffness matrix (Laplacian-like).
        M (csc_matrix): Mass matrix.
    """
    laplace = pcdlp_matrix(X, 2, nn=k, hs=2.0, rho=3.0, htype="ddr")
    M = sp.diags(np.ones(X.shape[0]))
    return laplace, M
    


def subspace_loss(U, V, m_diag):
    r"""Assuming the ground-truth eigenvectors are $M$-normalized ($u_j^\top M u_j=1$),
    the residual energy not captured by $\mathrm{span}(Y)$ is
    \begin{equation}
    r_j = \|u_j - YY^\top M u_j\|_M^2
        = 1 - \|Y^\top M u_j\|_2^2.
    \end{equation}
    We define the span loss as
    \begin{equation}
        \mathcal{L}_{\text{span}}
        = \sum_{j=1}^k \left(1 - \|Y^\top M u_j\|_2^2\right)
        = k - \|Y^\top M U_k\|_F^2 .
        \label{eq:span_loss}
    \end{equation}
    By construction, $\mathcal{L}_{\text{span}}=0$ implies $\mathrm{span}(U_k)\subseteq\mathrm{span}(Y)$.

    U: [N, d]
    V: [N, k]
    m_diag: [N, ]

    we compoute an average instead of a sum -- 1 - 1/k Sum[...]
    """
    # U: prediction Y, [N, d]
    # V: ground truth U_k, [N, k]
    # m_diag: M, [N]
    
    # MV = M @ V -> [N, k]
    MV = m_diag[..., None] * V
    
    # term = Y^T M U_k = U^T @ MV -> [d, k]
    term = np.matmul(U.T, MV)
    
    # squared Frobenius norm
    norm_sq = np.sum(term ** 2)
    
    k = V.shape[1]
    
    # Average loss: 1 - 1/k * ||Y^T M U_k||_F^2
    loss = 1.0 - (1.0 / k) * norm_sq
    return loss

def _best_match(pred: np.ndarray, gt: np.ndarray, mass: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.ndim != 2 or gt.ndim != 2:
        raise ValueError(f"pred and gt must be 2D, got {pred.shape}, {gt.shape}")
    if pred.shape[0] != gt.shape[0]:
        raise ValueError(f"N mismatch: pred {pred.shape} vs gt {gt.shape}")

    sim = np.abs(outer_cosine_similarity(pred, gt, M=mass))
    best_idx = np.argmax(sim, axis=0).astype(np.int64)
    best_score = sim[best_idx, np.arange(sim.shape[1])].astype(np.float32)

    if mass is None:
        dots = np.sum(pred[:, best_idx] * gt, axis=0)
    else:
        m = np.asarray(mass, dtype=np.float64).reshape(-1)
        dots = np.sum((m[:, None] * pred[:, best_idx]) * gt, axis=0)
    sign = np.where(dots >= 0.0, 1.0, -1.0).astype(np.float32)
    return best_idx, best_score, sign


def main():
    parser = ArgumentParser(description="Infer network eigens and compare with mesh/point-cloud eigens.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the point cloud data.")
    parser.add_argument("--glob", type=str, default="*", help="Glob pattern to match point cloud files.")
    parser.add_argument("--k", type=int, default=0, help="Number of modes used for pairwise comparison.")
    parser.add_argument("--no-mass", action='store_true')
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference (cuda or cpu).")
    args = parser.parse_args()

    # Load the pre-trained model
    model: PretrainTraining = PretrainTraining.load_from_checkpoint(args.ckpt, weights_only=False, strict=False)
    model = model.to(device=args.device, dtype=torch.float32)
    model.eval()

    arg_k = args.k if args.k > 0 else model.targ_dim_model

    print("💾 Model loaded from checkpoint:", args.ckpt)
    print(f"- Training Target Dim: {model.targ_dim}")
    print(f"- Model Target Dim: {model.targ_dim_model}")
    print(f"- Eigenvectors to compute: {arg_k}")
    print(f"- Model: {model.model.__class__.__name__}")
    print(f"- Config: {model.cfg}")
    # Load the point cloud data
    data_dir = Path(args.data_dir)
    folders = list(data_dir.glob(args.glob))
    folders.sort()
    print(f"Found {len(folders)} samples in {data_dir}")

    for i, case in enumerate(folders):
        if not case.is_dir():
            print(f"Warning: {case} is not a directory.")
            continue
        print(case)
        pc_file = case / "sample_points.npy"
        if not pc_file.exists():
            print(f"Warning: {pc_file} does not exist, pass (maybe a mesh scene?).")
            continue
        
        # Ensure output directories exist
        input_dir = case / "input"
        input_dir.mkdir(exist_ok=True)
        inferred_dir = case / "inferred"
        inferred_dir.mkdir(exist_ok=True)

        pc_raw = np.load(pc_file)

        # unit normalize
        pc = normalize_pc(pc_raw, 0)

        L, M = compute_surface_lbo(pc)
        if i == 0:
            print(len(pc), pc.shape, pc.dtype)
        mass = M.diagonal().astype(np.float32)
        L = sp.csr_matrix(L).astype(np.float64)
        M = sp.csr_matrix(M).astype(np.float64)

        pc_tensor = torch.tensor(pc, dtype=torch.float32, device=model.device).view(1, -1, 3)
        mass_tensor = torch.tensor(mass, dtype=torch.float32, device=model.device).view(1, -1, 1)

        mesh_evec_path = input_dir / "mesh_evec.npy"
        if not mesh_evec_path.exists():
            # Fallback for backward compatibility
            mesh_evec_path = case / "temp" / "mesh_evec.npy"
            if not mesh_evec_path.exists():
                 mesh_evec_path = case / "mesh_evec.npy"
        
        if not mesh_evec_path.exists():
            print(f"Warning: {mesh_evec_path} does not exist.")
            continue
        mesh_evec_full = np.load(mesh_evec_path)
        if mesh_evec_full.ndim != 2 or mesh_evec_full.shape[0] != pc.shape[0] or mesh_evec_full.shape[1] < arg_k:
            print(f"Warning: invalid mesh_evec shape: {mesh_evec_full.shape}.")
            continue
        mesh_evec = np.asarray(mesh_evec_full[:, :arg_k], dtype=np.float64)
        mesh_evec_tensor = torch.tensor(mesh_evec, dtype=torch.float32, device=model.device)

        pc_gev_start = perf_counter()
        pc_eval, pc_evec = solve_gev_ground_truth(L=L, M=M, k=arg_k)
        pc_gev_time = perf_counter() - pc_gev_start
        pc_eval = np.asarray(pc_eval, dtype=np.float64)
        pc_evec = np.asarray(pc_evec, dtype=np.float64)

        match_pc_to_mesh, score_mesh_pc, sign_pc_to_mesh = _best_match(pc_evec, mesh_evec, mass)

        pc_eval_aligned = pc_eval[match_pc_to_mesh]
        pc_evec_aligned = pc_evec[:, match_pc_to_mesh] * sign_pc_to_mesh.reshape(1, -1)

        np.save(input_dir / "points.npy", pc.astype(np.float32))
        np.save(input_dir / "mass.npy", mass.astype(np.float32))

        np.save(inferred_dir / "pc_eval.npy", pc_eval_aligned.astype(np.float32))
        np.save(inferred_dir / "pc_evec.npy", pc_evec_aligned.astype(np.float32))

        np.save(inferred_dir / "match_pc_to_mesh.npy", match_pc_to_mesh)
        np.save(inferred_dir / "sign_pc_to_mesh.npy", sign_pc_to_mesh)
        np.save(inferred_dir / "score_mesh_pc.npy", score_mesh_pc)
        precisions: dict[str, dict[str, dict[str, float] | dict[str, list[float]]]] = {}

        for precision_name, autocast_dtype in [("fp32", None), ("fp16", torch.float16)]:
            if autocast_dtype is not None and model.device.type != "cuda":
                autocast_dtype = None

            # Warmup...
            for _ in range(20):
                with torch.no_grad():
                    if autocast_dtype is None:
                        outputs, pred_original = model.forward(pc_tensor, mass_tensor, args.no_mass, return_time=False)
                    else:
                        with torch.autocast(device_type=model.device.type, dtype=autocast_dtype):
                            outputs, pred_original = model.forward(pc_tensor, mass_tensor, args.no_mass, return_time=False)
                    subspace_vector = outputs[0].float()
                solve_gev_from_subspace_cuda(
                    subspace_vector,
                    pc_tensor[0],
                    mass_tensor[0],
                    k=model.targ_dim_model,
                    L=L,
                    M=M,
                    precision="32",
                )

            start_time = perf_counter()
            with torch.no_grad():
                if autocast_dtype is None:
                    outputs, pred_original, infer_T = model.forward(pc_tensor, mass_tensor, args.no_mass, return_time=True)
                else:
                    with torch.autocast(device_type=model.device.type, dtype=autocast_dtype):
                        outputs, pred_original, infer_T = model.forward(pc_tensor, mass_tensor, args.no_mass, return_time=True)
                subspace_vector = outputs[0].float()
            if torch.cuda.is_available() and model.device.type == "cuda":
                torch.cuda.synchronize()
            forward_time = perf_counter() - start_time
            qr_time = forward_time - infer_T

            net_gev_start = perf_counter()
            net_eval_t, net_evec_t = solve_gev_from_subspace_cuda(
                subspace_vector,
                pc_tensor[0],
                mass_tensor[0],
                k=model.targ_dim_model,
                L=L,
                M=M,
                precision="32",
            )
            if torch.cuda.is_available() and model.device.type == "cuda":
                torch.cuda.synchronize()
            net_gev_time = perf_counter() - net_gev_start

            loss_val = subspace_loss(subspace_vector.detach().cpu().numpy(), pc_evec, mass)

            net_eval = net_eval_t.detach().cpu().numpy().astype(np.float64)
            net_evec = net_evec_t.detach().cpu().numpy().astype(np.float64)

            residual_ours = np.linalg.norm(L @ net_evec - M @ net_evec * net_eval.reshape(1, -1), axis=0)
            avg_residual = np.mean(residual_ours)

            low_prec_start = perf_counter()
            solve_gev_ground_truth(L, M, k=arg_k, tol=avg_residual)
            same_residual_time = perf_counter() - low_prec_start

            match_net_to_mesh, score_mesh_net, sign_net_to_mesh = _best_match(net_evec, mesh_evec, mass)
            match_net_to_pc, score_pc_net, sign_net_to_pc = _best_match(net_evec, pc_evec, mass)
            evals = net_eval[match_net_to_pc]
            eval_relerr = np.abs((evals - pc_eval) / (pc_eval + 1e-8))
            eval_relerr[0] = 0.0

            net_eval_aligned = net_eval[match_net_to_mesh]
            net_evec_aligned = net_evec[:, match_net_to_mesh] * sign_net_to_mesh.reshape(1, -1)
            pred_original_np = pred_original.detach().cpu().numpy().astype(np.float32)

            np.save(inferred_dir / f"net_eval_{precision_name}.npy", net_eval_aligned.astype(np.float32))
            np.save(inferred_dir / f"net_evec_{precision_name}.npy", net_evec_aligned.astype(np.float32))
            np.save(inferred_dir / f"net_pred_original_{precision_name}.npy", pred_original_np)

            np.save(inferred_dir / f"match_net_to_mesh_{precision_name}.npy", match_net_to_mesh)
            np.save(inferred_dir / f"sign_net_to_mesh_{precision_name}.npy", sign_net_to_mesh)
            np.save(inferred_dir / f"score_mesh_net_{precision_name}.npy", score_mesh_net)

            np.save(inferred_dir / f"match_net_to_pc_{precision_name}.npy", match_net_to_pc)
            np.save(inferred_dir / f"sign_net_to_pc_{precision_name}.npy", sign_net_to_pc)
            np.save(inferred_dir / f"score_pc_net_{precision_name}.npy", score_pc_net)

            if precision_name == "fp32":
                np.save(inferred_dir / "net_eval.npy", net_eval_aligned.astype(np.float32))
                np.save(inferred_dir / "net_evec.npy", net_evec_aligned.astype(np.float32))
                np.save(inferred_dir / "net_pred_original.npy", pred_original_np)
                np.save(inferred_dir / "match_net_to_mesh.npy", match_net_to_mesh)
                np.save(inferred_dir / "sign_net_to_mesh.npy", sign_net_to_mesh)
                np.save(inferred_dir / "score_mesh_net.npy", score_mesh_net)
                np.save(inferred_dir / "match_net_to_pc.npy", match_net_to_pc)
                np.save(inferred_dir / "sign_net_to_pc.npy", sign_net_to_pc)
                np.save(inferred_dir / "score_pc_net.npy", score_pc_net)

            print(f"[{precision_name}] ⌛️ Forward time: {forward_time:.4f} seconds")
            print(f"[{precision_name}] ⌛️ Network GEV time: {net_gev_time:.4f} seconds (total={net_gev_time + forward_time:.4f})")

            precisions[precision_name] = {
                "loss": float(loss_val.item()),
                "times": {
                    # Time for forward pass
                    "forward": float(infer_T),
                    # Time for Weighted-QR decomposition
                    "qr": float(qr_time),
                    # Time for Reyleigh-Ritz Procedure
                    "network_gev": float(net_gev_time),
                    # Time for arpack to converge to machine precision
                    "pointcloud_gev": float(pc_gev_time),
                    # Time for arpack to converge to same precision
                    "same_residual_gev": float(same_residual_time),
                },
                "scores": {
                    "mesh_vs_pointcloud": score_mesh_pc.tolist(),
                    "mesh_vs_network": score_mesh_net.tolist(),
                    "pointcloud_vs_network": score_pc_net.tolist(),
                    "eval_relerr": eval_relerr.tolist(),
                },
            }

        print(f"⌛️ Point-cloud GEV time: {pc_gev_time:.4f} seconds")
        print(f"Scores mesh↔pc: mean={float(score_mesh_pc.mean()):.4f}, median={float(np.median(score_mesh_pc)):.4f}")
        for precision_name in ["fp32", "fp16"]:
            if precision_name not in precisions:
                continue
            score_mesh_net = np.asarray(precisions[precision_name]["scores"]["mesh_vs_network"], dtype=np.float64)
            score_pc_net = np.asarray(precisions[precision_name]["scores"]["pointcloud_vs_network"], dtype=np.float64)
            print(f"[{precision_name}] Scores mesh↔net: mean={float(score_mesh_net.mean()):.4f}, median={float(np.median(score_mesh_net)):.4f}")
            print(f"[{precision_name}] Scores pc↔net: mean={float(score_pc_net.mean()):.4f}, median={float(np.median(score_pc_net)):.4f}")

        information_json = {
            "file_name": pc_file.absolute().as_posix(),
            "n_points": int(pc.shape[0]),
            "arg_k": int(arg_k),
            "net_k": int(model.targ_dim_model),
            "times": {
                "forward": float(precisions.get("fp32", precisions.get("fp16"))["times"]["forward"]),
                "qr": float(precisions.get("fp32", precisions.get("fp16"))["times"]["qr"]),
                "network_gev": float(precisions.get("fp32", precisions.get("fp16"))["times"]["network_gev"]),
                "pointcloud_gev": float(pc_gev_time),
                # Time for arpack to converge to same precision
                "same_residual_gev": float(precisions.get("fp32", precisions.get("fp16"))["times"]["same_residual_gev"]),
            },
            "scores": {
                "mesh_vs_pointcloud": score_mesh_pc.tolist(),
                "mesh_vs_network": precisions.get("fp32", precisions.get("fp16"))["scores"]["mesh_vs_network"],
                "pointcloud_vs_network": precisions.get("fp32", precisions.get("fp16"))["scores"]["pointcloud_vs_network"],
                "subspace_loss": precisions.get("fp32", precisions.get("fp16"))["loss"],
                "eval_relerr": precisions.get("fp32", precisions.get("fp16"))["scores"]["eval_relerr"],
            },
            "precisions": precisions,
        }
        with (inferred_dir / "results.json").open("w", encoding="utf-8") as f:
            json.dump(information_json, f, indent=4)


if __name__ == "__main__":
    main()
