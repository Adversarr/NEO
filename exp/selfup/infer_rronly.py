from time import perf_counter
from pathlib import Path
from argparse import ArgumentParser
import json

import numpy as np
import scipy.sparse as sp
import torch

from g2pt.training.selfup_rronly import SelfSupervisedRQTraining
from g2pt.utils.gev import outer_cosine_similarity, solve_gev_from_subspace_cuda, solve_gev_gt
from g2pt.utils.mesh_feats import point_cloud_laplacian
from g2pt.data.transforms import normalize_pc


def _best_match(pred: np.ndarray, gt: np.ndarray, mass: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find best matching pairs between predicted and ground truth eigenvectors."""
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
    args = parser.parse_args()

    # Load the pre-trained model
    model: SelfSupervisedRQTraining = SelfSupervisedRQTraining.load_from_checkpoint(args.ckpt, weights_only=False)
    model.eval()

    arg_k = args.k if args.k > 0 else model.targ_dim_model

    print("💾 Model loaded from checkpoint:", args.ckpt)
    print(f"- Training Target Dim: {model.targ_dim}")
    print(f"- Model Target Dim: {model.targ_dim_model}")
    print(f"- Eigenvectors to compute: {arg_k}")
    # Load the point cloud data
    data_dir = Path(args.data_dir)
    folders = list(data_dir.glob(args.glob))
    folders.sort()
    print(f"Found {len(folders)} samples in {data_dir}")

    for i, case in enumerate(folders):
        pc_file = case / "sample_points.npy"
        if not pc_file.exists():
            print(f"Warning: {pc_file} does not exist.")
            continue
        pc_raw = np.load(pc_file)

        # unit normalize
        pc = normalize_pc(pc_raw, 0)

        L, M = point_cloud_laplacian(pc)
        if i == 0:
            print(len(pc), pc.shape, pc.dtype)
        mass = M.diagonal().astype(np.float32)
        L = sp.csr_matrix(L).astype(np.float64)
        M = sp.csr_matrix(M).astype(np.float64)

        pc_tensor = torch.tensor(pc, dtype=torch.float32, device=model.device).view(1, -1, 3)
        mass_tensor = torch.tensor(mass, dtype=torch.float32, device=model.device).view(1, -1, 1)

        mesh_evec_path = case / "mesh_evec.npy"
        if not mesh_evec_path.exists():
            print(f"Warning: {mesh_evec_path} does not exist.")
            continue
        mesh_evec_full = np.load(mesh_evec_path)
        if mesh_evec_full.ndim != 2 or mesh_evec_full.shape[0] != pc.shape[0] or mesh_evec_full.shape[1] < arg_k:
            print(f"Warning: invalid mesh_evec shape: {mesh_evec_full.shape}.")
            continue
        mesh_evec = np.asarray(mesh_evec_full[:, :arg_k], dtype=np.float64)

        start_time = perf_counter()
        with torch.no_grad():
            outputs, pred_original = model.forward(pc_tensor, mass_tensor, return_original=True)
            subspace_vector = outputs[0]
        if torch.cuda.is_available() and model.device.type == "cuda":
            torch.cuda.synchronize()
        forward_time = perf_counter() - start_time

        net_gev_start = perf_counter()
        net_eval_t, net_evec_t = solve_gev_from_subspace_cuda(
            subspace_vector,
            pc_tensor[0],
            mass_tensor[0],
            k=model.targ_dim_model,
            L=L,
            M=M,
        )
        if torch.cuda.is_available() and model.device.type == "cuda":
            torch.cuda.synchronize()
        net_gev_time = perf_counter() - net_gev_start

        net_eval = net_eval_t.detach().cpu().numpy().astype(np.float64)
        net_evec = net_evec_t.detach().cpu().numpy().astype(np.float64)

        pc_gev_start = perf_counter()
        pc_eval, pc_evec = solve_gev_gt(pc_tensor[0], mass_tensor[0], k=arg_k, L=L, M=M)
        pc_gev_time = perf_counter() - pc_gev_start
        pc_eval = np.asarray(pc_eval, dtype=np.float64)
        pc_evec = np.asarray(pc_evec, dtype=np.float64)

        match_pc_to_mesh, score_mesh_pc, sign_pc_to_mesh = _best_match(pc_evec, mesh_evec, mass)
        match_net_to_mesh, score_mesh_net, sign_net_to_mesh = _best_match(net_evec, mesh_evec, mass)
        match_net_to_pc, score_pc_net, sign_net_to_pc = _best_match(net_evec, pc_evec, mass)

        np.save(case / "points.npy", pc.astype(np.float32))
        np.save(case / "mass.npy", mass.astype(np.float32))

        np.save(case / "pc_eval.npy", pc_eval.astype(np.float32))
        np.save(case / "pc_evec.npy", pc_evec.astype(np.float32))

        np.save(case / "net_eval.npy", net_eval.astype(np.float32))
        np.save(case / "net_evec.npy", net_evec.astype(np.float32))
        np.save(case / "net_pred_original.npy", pred_original.detach().cpu().numpy().astype(np.float32))

        np.save(case / "match_pc_to_mesh.npy", match_pc_to_mesh)
        np.save(case / "sign_pc_to_mesh.npy", sign_pc_to_mesh)
        np.save(case / "score_mesh_pc.npy", score_mesh_pc)

        np.save(case / "match_net_to_mesh.npy", match_net_to_mesh)
        np.save(case / "sign_net_to_mesh.npy", sign_net_to_mesh)
        np.save(case / "score_mesh_net.npy", score_mesh_net)

        np.save(case / "match_net_to_pc.npy", match_net_to_pc)
        np.save(case / "sign_net_to_pc.npy", sign_net_to_pc)
        np.save(case / "score_pc_net.npy", score_pc_net)

        print(f"⌛️ Forward time: {forward_time:.4f} seconds")
        print(f"⌛️ Network GEV time: {net_gev_time:.4f} seconds (total={net_gev_time + forward_time:.4f})")
        print(f"⌛️ Point-cloud GEV time: {pc_gev_time:.4f} seconds")
        print(f"Scores mesh↔pc: mean={float(score_mesh_pc.mean()):.4f}, median={float(np.median(score_mesh_pc)):.4f}")
        print(f"Scores mesh↔net: mean={float(score_mesh_net.mean()):.4f}, median={float(np.median(score_mesh_net)):.4f}")
        print(f"Scores pc↔net: mean={float(score_pc_net.mean()):.4f}, median={float(np.median(score_pc_net)):.4f}")

        information_json = {
            "file_name": pc_file.absolute().as_posix(),
            "n_points": int(pc.shape[0]),
            "arg_k": int(arg_k),
            "net_k": int(model.targ_dim_model),
            "times": {
                "forward": float(forward_time),
                "network_gev": float(net_gev_time),
                "pointcloud_gev": float(pc_gev_time),
            },
            "scores": {
                "mesh_vs_pointcloud": score_mesh_pc.tolist(),
                "mesh_vs_network": score_mesh_net.tolist(),
                "pointcloud_vs_network": score_pc_net.tolist(),
            },
        }
        with (case / "results.json").open("w", encoding="utf-8") as f:
            json.dump(information_json, f, indent=4)


if __name__ == "__main__":
    main()

