from pathlib import Path
from argparse import ArgumentParser
import json
from time import perf_counter
import numpy as np
import torch
from tqdm import tqdm
from g2pt.training.pretrain import PretrainTraining
from g2pt.utils.gev import outer_cosine_similarity, solve_gev_from_subspace_with_gt
from g2pt.data.datasets.shapenet_h5 import PreprocessedShapeNetDataModule, PreprocessedShapeNetDataModuleConfig

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
    if not isinstance(U, np.ndarray):
        U = U.detach().cpu().numpy()
    if not isinstance(V, np.ndarray):
        V = V.detach().cpu().numpy()
    if not isinstance(m_diag, np.ndarray):
        m_diag = m_diag.detach().cpu().numpy()


    # U: prediction Y, [N, d]
    # V: ground truth U_k, [N, k]
    # m_diag: M, [N]
    
    # MV = M @ V -> [N, k]
    MV = m_diag[..., None] * V
    
    # term = Y^T M U_k = U^T @ MV -> [d, k]
    term = np.matmul(U.T, MV)
    per_mode_error = term ** 2
    # squared Frobenius norm
    norm_sq = np.sum(per_mode_error)
    
    k = V.shape[1]
    
    # Average loss: 1 - 1/k * ||Y^T M U_k||_F^2
    loss = 1.0 - (1.0 / k) * norm_sq
    per_mode_energy = np.sum(per_mode_error, axis=0)
    per_mode_loss = 1.0 - per_mode_energy
    return loss, per_mode_loss

def main():
    parser = ArgumentParser(description="Infer network eigens and compare with mesh/point-cloud eigens.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--data_dir", type=str, default='ldata/preprocessed_shapenet_h5', help="Directory containing the point cloud data.")
    parser.add_argument("--glob", type=str, default="*", help="Glob pattern to match point cloud files.")
    parser.add_argument("--k", type=int, default=0, help="Number of modes used for pairwise comparison.")
    parser.add_argument("--precision", type=str, choices=["32", "64"], default="64", help="Precision for GEV solver (32 or 64).")
    parser.add_argument("--infer_precision", type=str, choices=["32", "16"], default="32", help="Precision for inference (32 or 16).")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference (cuda or cpu).")
    parser.add_argument("--sample_seed", type=int, default=42, help="Seed for random sampling of point clouds.")
    parser.add_argument("--sample_count", type=int, default=2048, help="Number of sampling")
    parser.add_argument("--limit", type=int, default=1024, help="Number of validation.")
    parser.add_argument('--out', type=str, default='outputs.json')
    parser.add_argument('--noise_strength', type=float, default=0.0)
    args = parser.parse_args()

    # Load the pre-trained model
    model: PretrainTraining = PretrainTraining.load_from_checkpoint(args.ckpt, weights_only=False, strict=False)
    shapenet_preprocessed = args.data_dir
    dataloader = PreprocessedShapeNetDataModule(cfg=PreprocessedShapeNetDataModuleConfig(
        batch_size=1,
        data_dir=shapenet_preprocessed,
        split_ratio=0.9, # match Train
        targ_dim=model.targ_dim,
        seed=42, # match Train
    )).val_dataloader_with_mesh()
    train_dl = PreprocessedShapeNetDataModule(cfg=PreprocessedShapeNetDataModuleConfig(
        batch_size=1,
        data_dir=shapenet_preprocessed,
        split_ratio=0.9, # match Train
        targ_dim=model.targ_dim,
        seed=42, # match Train
    )).train_dataloader()
    print(f"Validation set size: {len(dataloader)} vs. Training set size: {len(train_dl)}")

    infer_precision = torch.float32 if args.infer_precision == "32" else torch.float16
    model = model.to(device=args.device, dtype=infer_precision)
    model.eval()

    arg_k = args.k if args.k > 0 else model.targ_dim_model

    print("💾 Model loaded from checkpoint:", args.ckpt)
    print(f"- Training Target Dim: {model.targ_dim}")
    print(f"- Model Target Dim: {model.targ_dim_model}")
    print(f"- Eigenvectors to compute: {arg_k}")
    results = []

    total = min(args.limit, len(dataloader))
    for i, case in enumerate(tqdm(dataloader, total=total, desc="Validating")):
        if i >= args.limit:
            break
        x = case["points"].to(model.device)
        mass = case["mass"].to(model.device)
        y = case['evecs'].to(model.device)
        if args.noise_strength > 0.0:
            x = x + args.noise_strength * torch.randn_like(x)
        start = perf_counter()
        with torch.no_grad():
            outputs, _pred_original = model.forward(x.to(model.dtype), mass.to(model.dtype))
        torch.cuda.synchronize()
        eval_time = perf_counter() - start
        outputs = outputs.float()

        subloss, per_mode_loss = subspace_loss(
            outputs[0].detach().cpu().numpy(),
            y[0].detach().cpu().numpy(),
            mass[0].detach().cpu().numpy().flatten(),
        )

        m0 = mass[0].detach().cpu().numpy()  # Convert mass to numpy for GEVP.

        evec, evec_gt, evals, evals_gt = solve_gev_from_subspace_with_gt(
            outputs[0], x[0], mass[0], k=arg_k, return_evals=True
        )  # [n, c], [n, c]
        scores = np.abs(outer_cosine_similarity(evec, evec_gt, M=m0))  # (c, d)
        max_scores = np.max(np.abs(scores), axis=0)  # [d]
        idx = np.argmax(scores, axis=0) # [d]
        evals = evals[idx]

        eval_relerr = np.abs((evals - evals_gt) / (evals_gt + 1e-8))
        eval_relerr[0] = 0.0

        information_json = {
            "n_points": int(x.shape[1]),
            "arg_k": int(arg_k),
            "net_k": int(model.targ_dim_model),
            "scores": {
                "pointcloud_vs_network": max_scores.tolist(),
            },
            "eval_relerr": eval_relerr.tolist(),
            "subloss": float(subloss),
            "per_mode_loss": per_mode_loss.tolist(),
            "eval_time": float(eval_time),
        }
        results.append(information_json)

    Path(args.out).write_text(json.dumps(results))

if __name__ == "__main__":
    main()
