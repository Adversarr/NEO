from time import perf_counter
from pathlib import Path
from argparse import ArgumentParser
import json

import numpy as np
import scipy.sparse as sp
import torch
import trimesh

from g2pt.training.pretrain import PretrainTraining
from g2pt.utils.gev import outer_cosine_similarity, solve_gev_from_subspace_cuda, solve_gev_ground_truth
from g2pt.utils.mesh_feats import mesh_laplacian
from g2pt.data.transforms import normalize_pc

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


def _load_any_mesh(path: Path) -> tuple[trimesh.Trimesh | trimesh.PointCloud, np.ndarray, np.ndarray | None]:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values()]
        if not geoms:
            raise ValueError(f"No geometry found in {path}")
        meshes = []
        pcs = []
        for g in geoms:
            if isinstance(g, trimesh.Trimesh):
                if g.vertices.size > 0 and g.faces.size > 0:
                    meshes.append(g)
            elif isinstance(g, trimesh.PointCloud):
                if g.vertices.size > 0:
                    pcs.append(g)
        if meshes:
            merged = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
            vertices = np.asarray(merged.vertices, dtype=np.float32)
            faces = np.asarray(merged.faces, dtype=np.int64)
            return merged, vertices, faces
        if pcs:
            merged_pc = trimesh.points.PointCloud(vertices=np.concatenate([np.asarray(pc.vertices) for pc in pcs], axis=0))
            vertices = np.asarray(merged_pc.vertices, dtype=np.float32)
            return merged_pc, vertices, None
        raise ValueError(f"No supported geometry found in {path}")

    if isinstance(loaded, trimesh.Trimesh):
        vertices = np.asarray(loaded.vertices, dtype=np.float32)
        faces = np.asarray(loaded.faces, dtype=np.int64) if loaded.faces is not None and len(loaded.faces) > 0 else None
        return loaded, vertices, faces

    if isinstance(loaded, trimesh.PointCloud):
        vertices = np.asarray(loaded.vertices, dtype=np.float32)
        return loaded, vertices, None

    raise ValueError(f"Unsupported trimesh type: {type(loaded)} for {path}")


def _safe_case_name(path: Path) -> str:
    name = path.stem
    name = name.replace(" ", "_")
    return name


def main():
    parser = ArgumentParser(description="Infer network eigens and compare with mesh/point-cloud eigens.")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to the model checkpoint.")
    parser.add_argument("--model_path", type=str, default=None, help="Alias of --ckpt.")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing the mesh files.")
    parser.add_argument("--input_dir", type=str, default=None, help="Alias of --data_dir.")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--output_dir", type=str, default=None, help="Alias of --out_dir.")
    parser.add_argument("--glob", type=str, default="*", help="Glob pattern to match mesh files.")
    parser.add_argument("--k", type=int, default=0, help="Number of modes used for pairwise comparison.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference (cuda or cpu).")
    args = parser.parse_args()

    ckpt_path = args.ckpt or args.model_path
    data_dir = args.data_dir or args.input_dir
    out_dir = args.out_dir or args.output_dir
    missing = []
    if not ckpt_path:
        missing.append("--ckpt/--model_path")
    if not data_dir:
        missing.append("--data_dir/--input_dir")
    if not out_dir:
        missing.append("--out_dir/--output_dir")
    if missing:
        parser.error(f"Missing required arguments: {', '.join(missing)}")

    # Load the pre-trained model
    model: PretrainTraining = PretrainTraining.load_from_checkpoint(ckpt_path, weights_only=False, strict=False)
    model = model.to(device=args.device, dtype=torch.float32)
    model.eval()

    arg_k = args.k if args.k > 0 else model.targ_dim_model

    print("💾 Model loaded from checkpoint:", ckpt_path)
    print(f"- Training Target Dim: {model.targ_dim}")
    print(f"- Model Target Dim: {model.targ_dim_model}")
    print(f"- Eigenvectors to compute: {arg_k}")
    print(f"- Model: {model.model.__class__.__name__}")
    print(f"- Config: {model.cfg}")

    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = list(data_dir.glob(args.glob))
    candidates.sort()
    mesh_files = [p for p in candidates if p.is_file() and p.suffix.lower() in [".ply", ".obj", ".off"]]
    print(f"Found {len(mesh_files)} meshes in {data_dir}")

    for i, mesh_path in enumerate(mesh_files):
        case_name = _safe_case_name(mesh_path)
        case_dir = out_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)

        input_dir = case_dir / "input"
        input_dir.mkdir(exist_ok=True)
        inferred_dir = case_dir / "inferred"
        inferred_dir.mkdir(exist_ok=True)

        try:
            loaded, vertices_raw, faces_raw = _load_any_mesh(mesh_path)
        except Exception as e:
            print(f"Warning: failed to load {mesh_path}: {e}")
            continue

        original_ply_path = case_dir / "original.ply"
        try:
            vertices = np.asarray(loaded.vertices, dtype=np.float32)
            vertices = vertices - np.mean(vertices, axis=0, keepdims=True)
            max_norm = np.max(np.linalg.norm(vertices, axis=1, keepdims=True))
            vertices = vertices / max_norm
            trimesh.Trimesh(vertices=vertices, faces=faces_raw, process=False).export(str(original_ply_path))
            # loaded.export(str(original_ply_path))
        except Exception:
            if faces_raw is not None and faces_raw.size > 0:
                trimesh.Trimesh(vertices=vertices_raw, faces=faces_raw, process=False).export(str(original_ply_path))
            else:
                trimesh.PointCloud(vertices=vertices_raw).export(str(original_ply_path))

        vertices = normalize_pc(vertices_raw, 0)
        faces = faces_raw

        np.save(input_dir / "points.npy", vertices.astype(np.float32))
        if faces is not None:
            np.save(input_dir / "faces.npy", faces.astype(np.int64))

        if faces is None or faces.size == 0:
            print(f"Warning: {mesh_path} has no faces, skip mesh Laplacian inference.")
            continue

        lap_start = perf_counter()
        L, M = mesh_laplacian(vertices.astype(np.float64), faces.astype(np.int64))
        lap_time = perf_counter() - lap_start

        mass = sp.csr_matrix(M).diagonal().astype(np.float32)
        L = sp.csr_matrix(L).astype(np.float64)
        M = sp.csr_matrix(M).astype(np.float64)

        mesh_tensor = torch.tensor(vertices, dtype=torch.float32, device=model.device).view(1, -1, 3)
        mass_tensor = torch.tensor(mass, dtype=torch.float32, device=model.device).view(1, -1, 1)

        mesh_gev_start = perf_counter()
        mesh_eval, mesh_evec = solve_gev_ground_truth(L=L, M=M, k=arg_k)
        mesh_gev_time = perf_counter() - mesh_gev_start
        mesh_eval = np.asarray(mesh_eval, dtype=np.float64)
        mesh_evec = np.asarray(mesh_evec, dtype=np.float64)

        np.save(input_dir / "mass.npy", mass.astype(np.float32))
        np.save(inferred_dir / "mesh_eval.npy", mesh_eval.astype(np.float32))
        np.save(inferred_dir / "mesh_evec.npy", mesh_evec.astype(np.float32))

        precisions: dict[str, dict[str, dict[str, float] | dict[str, list[float]]]] = {}

        for precision_name, autocast_dtype in [("fp32", None), ("fp16", torch.float16)]:
            if autocast_dtype is not None and model.device.type != "cuda":
                autocast_dtype = None

            for _ in range(20):
                with torch.no_grad():
                    if autocast_dtype is None:
                        outputs, pred_original = model.forward(mesh_tensor, mass_tensor, return_time=False)
                    else:
                        with torch.autocast(device_type=model.device.type, dtype=autocast_dtype):
                            outputs, pred_original = model.forward(mesh_tensor, mass_tensor, return_time=False)
                    subspace_vector = outputs[0].float()
                solve_gev_from_subspace_cuda(
                    subspace_vector,
                    mesh_tensor[0],
                    mass_tensor[0],
                    k=model.targ_dim_model,
                    L=L,
                    M=M,
                    precision="32",
                )

            start_time = perf_counter()
            with torch.no_grad():
                if autocast_dtype is None:
                    outputs, pred_original, infer_T = model.forward(mesh_tensor, mass_tensor, return_time=True)
                else:
                    with torch.autocast(device_type=model.device.type, dtype=autocast_dtype):
                        outputs, pred_original, infer_T = model.forward(mesh_tensor, mass_tensor, return_time=True)
                subspace_vector = outputs[0].float()
            if torch.cuda.is_available() and model.device.type == "cuda":
                torch.cuda.synchronize()
            forward_time = perf_counter() - start_time
            qr_time = forward_time - infer_T

            net_gev_start = perf_counter()
            net_eval_t, net_evec_t = solve_gev_from_subspace_cuda(
                subspace_vector,
                mesh_tensor[0],
                mass_tensor[0],
                k=model.targ_dim_model,
                L=L,
                M=M,
                precision="32",
            )
            if torch.cuda.is_available() and model.device.type == "cuda":
                torch.cuda.synchronize()
            net_gev_time = perf_counter() - net_gev_start

            loss_val = subspace_loss(subspace_vector.detach().cpu().numpy(), mesh_evec, mass)

            net_eval = net_eval_t.detach().cpu().numpy().astype(np.float64)
            net_evec = net_evec_t.detach().cpu().numpy().astype(np.float64)

            residual_ours = np.linalg.norm(L @ net_evec - M @ net_evec * net_eval.reshape(1, -1), axis=0)
            avg_residual = np.mean(residual_ours)

            low_prec_start = perf_counter()
            solve_gev_ground_truth(L, M, k=arg_k, tol=avg_residual)
            same_residual_time = perf_counter() - low_prec_start

            match_net_to_mesh, score_mesh_net, sign_net_to_mesh = _best_match(net_evec, mesh_evec, mass)
            evals = net_eval[match_net_to_mesh]
            eval_relerr = np.abs((evals - mesh_eval) / (mesh_eval + 1e-8))
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

            if precision_name == "fp32":
                np.save(inferred_dir / "net_eval.npy", net_eval_aligned.astype(np.float32))
                np.save(inferred_dir / "net_evec.npy", net_evec_aligned.astype(np.float32))
                np.save(inferred_dir / "net_pred_original.npy", pred_original_np)
                np.save(inferred_dir / "match_net_to_mesh.npy", match_net_to_mesh)
                np.save(inferred_dir / "sign_net_to_mesh.npy", sign_net_to_mesh)
                np.save(inferred_dir / "score_mesh_net.npy", score_mesh_net)

            print(f"[{i+1}/{len(mesh_files)}] {mesh_path.name} [{precision_name}] ⌛️ Forward time: {forward_time:.4f} seconds")
            print(
                f"[{i+1}/{len(mesh_files)}] {mesh_path.name} [{precision_name}] ⌛️ Network GEV time: {net_gev_time:.4f} seconds (total={net_gev_time + forward_time:.4f})"
            )

            precisions[precision_name] = {
                "loss": float(loss_val.item()),
                "times": {
                    "forward": float(infer_T),
                    "qr": float(qr_time),
                    "network_gev": float(net_gev_time),
                    "mesh_laplacian": float(lap_time),
                    "mesh_gev": float(mesh_gev_time),
                    "same_residual_gev": float(same_residual_time),
                },
                "scores": {
                    "mesh_vs_network": score_mesh_net.tolist(),
                    "eval_relerr": eval_relerr.tolist(),
                },
            }

        print(f"[{i+1}/{len(mesh_files)}] {mesh_path.name} ⌛️ Mesh Laplacian time: {lap_time:.4f} seconds")
        print(f"[{i+1}/{len(mesh_files)}] {mesh_path.name} ⌛️ Mesh GEV time: {mesh_gev_time:.4f} seconds")
        for precision_name in ["fp32", "fp16"]:
            if precision_name not in precisions:
                continue
            score_mesh_net = np.asarray(precisions[precision_name]["scores"]["mesh_vs_network"], dtype=np.float64)
            print(f"[{i+1}/{len(mesh_files)}] {mesh_path.name} [{precision_name}] Scores mesh↔net: mean={float(score_mesh_net.mean()):.4f}, median={float(np.median(score_mesh_net)):.4f}")

        information_json = {
            "file_name": mesh_path.absolute().as_posix(),
            "n_points": int(vertices.shape[0]),
            "n_verts": int(vertices.shape[0]),
            "n_faces": int(faces.shape[0]) if faces is not None else 0,
            "arg_k": int(arg_k),
            "net_k": int(model.targ_dim_model),
            "times": {
                "mesh_laplacian": float(lap_time),
                "mesh_gev": float(mesh_gev_time),
                "forward": float(precisions.get("fp32", precisions.get("fp16"))["times"]["forward"]),
                "qr": float(precisions.get("fp32", precisions.get("fp16"))["times"]["qr"]),
                "network_gev": float(precisions.get("fp32", precisions.get("fp16"))["times"]["network_gev"]),
                "same_residual_gev": float(precisions.get("fp32", precisions.get("fp16"))["times"]["same_residual_gev"]),
            },
            "scores": {
                "mesh_vs_network": precisions.get("fp32", precisions.get("fp16"))["scores"]["mesh_vs_network"],
                "subspace_loss": precisions.get("fp32", precisions.get("fp16"))["loss"],
                "eval_relerr": precisions.get("fp32", precisions.get("fp16"))["scores"]["eval_relerr"],
            },
            "precisions": precisions,
        }
        with (inferred_dir / "results.json").open("w", encoding="utf-8") as f:
            json.dump(information_json, f, indent=4)


if __name__ == "__main__":
    main()
