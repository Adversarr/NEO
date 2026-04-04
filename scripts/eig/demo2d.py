#!/usr/bin/env python3
"""
Demo 2d. works now.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float32)

def build_1d_fem(n, k, final_dim=2):
    """
    1D linear element stiffness L1 and consistent mass M1 (natural boundary).
    Returns L1, M1, mesh point coordinates x (n,), mesh step size h
    """
    assert n >= 2
    h = 1.0 / (n - 1)
    L = torch.zeros((n, n))
    for i in range(n - 1):
        L[i, i] += 1.0 / h
        L[i, i + 1] += -1.0 / h
        L[i + 1, i] += -1.0 / h
        L[i + 1, i + 1] += 1.0 / h
    L /= k ** (1 / final_dim)
    M = torch.zeros((n, n))
    for i in range(n - 1):
        M[i, i] += 2.0 * h / 6.0
        M[i, i + 1] += 1.0 * h / 6.0
        M[i + 1, i] += 1.0 * h / 6.0
        M[i + 1, i + 1] += 2.0 * h / 6.0
    M *= k ** (1 / final_dim)
    x = torch.linspace(0.0, 1.0, n)
    return L, M, x, h

def build_2d_fem(n, delta, k):
    """
    2D Q1 tensor product FEM, rectangular mesh [0,1]x[0,1], natural (Neumann) boundary.
    Using tensor product formula:
      L2 = kron(Lx, My) + kron(Mx, Ly)
      M2 = kron(Mx, My)
    A = L2 + delta * M2
    Returns:
      X, Y: 2D mesh coordinates (ny, nx)
      L2, M2, w_lump(=M row sum), A
    """
    # Same direction: n points => nx=ny=n
    Lx, Mx, x, hx = build_1d_fem(n, k)
    Ly, My, y, hy = build_1d_fem(n, k)
    # Tensor product
    L2 = torch.kron(Lx, My) + torch.kron(Mx, Ly)
    M2 = torch.kron(Mx, My)
    A = L2 + delta * M2
    w_lump = M2.sum(dim=1)  # (N,)
    X, Y = torch.meshgrid(y, x, indexing="ij")  # (ny, nx)
    return X, Y, L2, M2, w_lump, A

def sample_b(batch_size, w_lump, device, scale_to_vol=True):
    """
    Load sampling: b_i = z_i / sqrt(w_i), w_i is lumped mass (approximates L2 isotropy: Cov(b)≈M^{-1})
    Optional scaling: makes E[b^T M b] ≈ vol, ensures resolution independence
    """
    n = w_lump.shape[0]
    z = torch.randn(batch_size, n, device=device)
    b = z / torch.sqrt(w_lump.to(device).unsqueeze(0))
    return b

def m_column_norms(E, M, eps=1e-12):
    """
    Column M-norm: d_i = sqrt(e_i^T M e_i), returns Q = E / d_i and d
    Uses .detach() to stabilize early training (avoids backprop through column norms)
    """
    ME = M @ E
    d2 = (E * ME).sum(dim=0)
    d = torch.sqrt(torch.clamp(d2, min=eps)).detach()
    Q = E / d.unsqueeze(0)
    return Q, d

def sol_apply(E, H_chol, M, b):
    """
    Subspace solver: Q = M-normalized basis; x̂ = Q H Q^T M b, where H = H_chol H_chol^T + 1e-8 I
    Input:
      E: (N,k), H_chol: (k,k), M: (N,N), b: (B,N) or (N,)
    Returns:
      x̂: (B,N) or (N,), Q: (N,k)
    """
    Q, _ = m_column_norms(E, M)
    H = H_chol
    # H = H_chol @ H_chol.t() + torch.eye(H_chol.shape[0], device=H_chol.device) * 1e-8
    if b.dim() == 1:
        Mb = M @ b
        y = (Mb.unsqueeze(0) @ Q).squeeze(0)
        z = y @ H
        x = z @ Q.t()
        return x, Q
    else:
        Mb = b @ M.t()
        y = Mb @ Q
        z = y @ H
        x = z @ Q.t()
        return x, Q

def rayleigh_ritz_in_subspace(E_np, A_np, M_np, k):
    """
    Subspace Ritz: solve (E^T A E) c = λ (E^T M E) c
    Returns λ(k,) and corresponding ambient vectors V = E c (N,k), M-normalized
    """
    K = E_np.T @ A_np @ E_np
    S = E_np.T @ M_np @ E_np
    S_ch = np.linalg.cholesky(S)
    S_inv = np.linalg.inv(S_ch)
    C = S_inv @ K @ S_inv.T
    w, U = np.linalg.eigh(C)
    Cvecs = S_inv.T @ U
    V = E_np @ Cvecs
    for j in range(k):
        nrm = np.sqrt(V[:, j].T @ M_np @ V[:, j])
        if nrm > 0:
            V[:, j] /= nrm
        if V[0, j] < 0:
            V[:, j] *= -1
    return w, V

def true_eigs(A_np, M_np, k):
    """
    True generalized eigenpairs: solve A v = λ M v (for evaluation)
    """
    M_ch = np.linalg.cholesky(M_np)
    Minv = np.linalg.inv(M_ch)
    C = Minv @ A_np @ Minv.T
    w, U = np.linalg.eigh(C)
    idx = np.argsort(w)
    w = w[idx][:k]
    U = U[:, idx[:k]]
    V = np.linalg.solve(M_ch.T, U)
    for j in range(k):
        nrm = np.sqrt(V[:, j].T @ M_np @ V[:, j])
        if nrm > 0:
            V[:, j] /= nrm
        if V[0, j] < 0:
            V[:, j] *= -1
    return w, V

def main():
    parser = argparse.ArgumentParser(description="2D heat problem subspace demo (scale-invariant)")
    parser.add_argument("--n", type=int, default=32, help="grid points per axis (N = n*n)")
    parser.add_argument("--k", type=int, default=8, help="subspace dimension")
    parser.add_argument("--epochs", type=int, default=2000, help="training epochs")
    parser.add_argument("--batch", type=int, default=32, help="mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-2, help="initial learning rate")
    parser.add_argument("--delta", type=float, default=10, help="shift A = L + delta*M")
    parser.add_argument("--comp-weight", type=float, default=1.0, help="weight of compliance")
    parser.add_argument("--kkt-weight", type=float, default=1.0, help="weight of energy term")
    parser.add_argument("--plot-every", type=int, default=200, help="print interval")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"], help="device")
    parser.add_argument("--savefig", type=str, default="", help="path to save final figure")
    parser.add_argument("--no-show", action="store_true", help="disable interactive plot")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="gradient clip norm")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    # 2D FEM
    X, Y, L, M, w_lump, A = build_2d_fem(args.n, args.delta, args.k)
    X, Y, L, M, A, w_lump = X.to(device), Y.to(device), L.to(device), M.to(device), A.to(device), w_lump.to(device)
    N = args.n * args.n
    vol = w_lump.sum().item()  # approximately |Ω| = 1

    # Trainable parameters
    E = nn.Parameter(torch.randn(N, args.k, device=device) / np.sqrt(N))
    H_chol = nn.Parameter(torch.eye(args.k, device=device) * 0.5 + 0.01 * torch.randn(args.k, args.k, device=device))
    params = [E, H_chol]

    optimizer = optim.Adam(params, lr=args.lr)
    gamma = 0.01 ** (1.0 / max(args.epochs, 1))  # Final LR=1e-2*initial
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    total_hist, comp_hist, kkt_hist, lr_hist, norm_hist = [], [], [], [], []

    # Evaluate true values (note: slow for large N, recommend n not too large)
    A_np = A.detach().cpu().numpy()
    M_np = M.detach().cpu().numpy()
    true_vals, true_vecs = true_eigs(A_np, M_np, args.k)
    print("True smallest generalized eigenvalues:", true_vals)

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()

        # Sample load (resolution-invariant scale)
        b = sample_b(args.batch, w_lump, device, scale_to_vol=True)  # (B,N)

        # Subspace solution x̂
        x_hat, Q = sol_apply(E, H_chol, M, b)  # (B,N)

        Ax_hat = x_hat @ A.t()
        Mb = b @ M.t()

        # Compliance: - E[(b^T M x̂)/vol]
        comp = - ((Mb * x_hat).sum(dim=1) / vol).mean()

        # Subspace energy (more stable): E[(0.5 x̂^T A x̂ - x̂^T M b)/vol]
        kkt = (0.5 * (Ax_hat * x_hat).sum(dim=1) - (Mb * x_hat).sum(dim=1)).mean() / vol

        # Asymmetric M-orthogonalization penalty (lower triangular + left detach)
        ME = M @ E
        ortho = torch.tril(E.t().detach() @ ME / vol, diagonal=1)   # Only penalize cross terms i>j
        norm_E = (E * ME).sum(dim=0) / vol                          # Column M-norm
        norm_pen = ortho.square().mean() + norm_E.mean()            # Light weight to prevent degeneracy and decouple solutions

        loss = args.comp_weight * comp + args.kkt_weight * kkt + 1e-8 * norm_pen
        loss.backward()

        torch.nn.utils.clip_grad_norm_(params, max_norm=args.grad_clip_norm)
        optimizer.step()
        scheduler.step()

        total_hist.append(loss.detach().cpu().item())
        comp_hist.append(comp.detach().cpu().item())
        kkt_hist.append(kkt.detach().cpu().item())
        lr_hist.append(optimizer.param_groups[0]["lr"])
        norm_hist.append(norm_pen.detach().cpu().item())

        if epoch % args.plot_every == 0 or epoch == 1 or epoch == args.epochs:
            with torch.no_grad():
                E_np = E.detach().cpu().numpy()
                rr_vals, rr_vecs = rayleigh_ritz_in_subspace(E_np, A_np, M_np, args.k)
            print(f"Epoch {epoch:4d} | lr {optimizer.param_groups[0]['lr']:.3e} "
                  f"| total {np.mean(total_hist[-min(args.plot_every, len(total_hist)):]):.3e} "
                  f"| comp {np.mean(comp_hist[-min(args.plot_every, len(comp_hist)):]):.3e} "
                  f"| kkt {np.mean(kkt_hist[-min(args.plot_every, len(kkt_hist)):]):.3e} "
                  f"| norm {np.mean(norm_hist[-min(args.plot_every, len(norm_hist)):]):.3e}")
            print("  Ritz eigvals (learned):", rr_vals)
            print("  True eigvals:          ", true_vals)

    # Final evaluation and visualization
    with torch.no_grad():
        E_np = E.detach().cpu().numpy()
        rr_vals, rr_vecs = rayleigh_ritz_in_subspace(E_np, A_np, M_np, args.k)

    fig, axs = plt.subplots(2, 2, figsize=(11, 9))
    # Loss curves
    axs[0,0].plot(total_hist, label="total")
    axs[0,0].plot(comp_hist, label="compliance")
    axs[0,0].plot(kkt_hist, label="energy")
    axs[0,0].set_title("Training losses")
    axs[0,0].set_xlabel("epoch")
    axs[0,0].legend()

    # Learning rate
    axs[0,1].plot(lr_hist)
    axs[0,1].set_title("Exponential LR")
    axs[0,1].set_xlabel("epoch")

    # Display first min(k,6) eigenfunctions: true vs learned (Ritz)
    kshow = min(args.k, 6)
    ny, nx = args.n, args.n
    # Create two grids: true vs learned
    true_grid = np.zeros((kshow, ny, nx), dtype=np.float32)
    rr_grid   = np.zeros((kshow, ny, nx), dtype=np.float32)
    for j in range(kshow):
        v_true = true_vecs[:, j]
        v_rr   = rr_vecs[:, j]
        if np.dot(v_true, v_rr) < 0:
            v_rr *= -1.0
        true_grid[j] = v_true.reshape(ny, nx)
        rr_grid[j]   = v_rr.reshape(ny, nx)

    im0 = axs[1,0].imshow(true_grid[2], origin="lower", extent=[0,1,0,1], cmap="viridis")
    axs[1,0].set_title("True eigenfunction e5")
    plt.colorbar(im0, ax=axs[1,0], fraction=0.046, pad=0.04)
    im1 = axs[1,1].imshow(rr_grid[2], origin="lower", extent=[0,1,0,1], cmap="viridis")
    axs[1,1].set_title("Learned (Ritz) eigenfunction e5")
    plt.colorbar(im1, ax=axs[1,1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if args.savefig:
        plt.savefig(args.savefig, dpi=160)
        print(f"Saved figure to {args.savefig}")
    if not args.no_show:
        plt.show()

if __name__ == "__main__":
    main()
