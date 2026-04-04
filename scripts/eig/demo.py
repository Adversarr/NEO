#!/usr/bin/env python3
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Usage: python tinyexp/demo.py --k=16 --epochs=15000 --savefig="demo.png" --lr=1e-1 --device=cuda --delta=10 --batch=64 --n=512
# Problems:
# 1. delta should be large enough to make the training stable, and converge to final solution.
# 2. original kkt loss does not work better than the new version.

#? Must be float32, we will use bfloat16 in the neural network, float32 is still more accurate
#? than real application
torch.set_default_dtype(torch.float32)

def build_1d_fem(n, delta, k):
    """
    Build 1D linear FEM stiffness (L) and mass (M) on [0,1] with uniform grid.
    Natural (Neumann) boundary conditions. A = L + delta*M is SPD.
    Returns:
      x: grid points (n,)
      L: stiffness (n,n)
      M: mass (n,n) (consistent mass)
      w_lump: lumped mass weights (row sums of M) (n,)
      A: L + delta*M
    """
    assert n >= 2
    h = 1.0 / (n - 1)

    # We use the discrete version of the stiffness matrix L
    # and the consistent mass matrix M. but I do not know why this is better than the physically correct one.

    # Assemble stiffness L for linear elements
    L = torch.zeros((n, n))
    for i in range(n - 1):
        L[i, i] += 1.0
        L[i, i + 1] += -1.0
        L[i + 1, i] += -1.0
        L[i + 1, i + 1] += 1.0
    L /= h * k
    # 1. scale by k: the k-smallest eigenvalue is ~k**2. we also scale by k in mass matrix.
    # 2. This ensures if i use k = 4, k = 16, the upper bound of L's k-smallest eigenvalue is ~k**2 => scale does not change
    # Our goal is to learn the eigenfunctions, not the SOL operator, this change is ok.

    # Consistent mass M: (h/6)*[[2,1],[1,2]] per element
    M = torch.zeros((n, n))
    for i in range(n - 1):
        M[i, i] += 2.0 / 6.0
        M[i, i + 1] += 1.0 / 6.0
        M[i + 1, i] += 1.0/ 6.0
        M[i + 1, i + 1] += 2.0 / 6.0
    M *= h * delta * k # Make the body heavier.

    w_lump = M.sum(dim=1)
    A = L + M
    x = torch.linspace(0.0, 1.0, n)
    return x, L, M, w_lump, A

def sample_b(batch_size, w_lump, device, scale_to_vol=True):
    """
    Sample loads b with covariance ~ M^{-1} using lumped mass:
      b_i = z_i / sqrt(w_i), z_i ~ N(0,1).
    Optionally scale b so that E[b^T M b] ≈ vol (= sum w_i), which makes compliance per unit volume resolution-invariant.
    """
    n = w_lump.shape[0]
    z = torch.randn(batch_size, n, device=device)
    b = z * torch.sqrt(w_lump.to(device).unsqueeze(0))  # Cov(b) ≈ M
    if scale_to_vol and False:
        #! If enabled, the scale is not correct anymore. (It only change the value of optimal)
        #! as we enlarge n, the optimal value of our minimizer will be smaller.
        vol = w_lump.sum().item()
        # E[b^T M b] = tr(M Cov(b)) = tr(I) = n; scale by sqrt(vol/n)
        b = b * np.sqrt(vol / n)
    return b

def m_column_norms(E, M, eps=1e-12):
    """
    Compute M-norms of columns of E: d_i = sqrt(e_i^T M e_i) + eps.
    Return Q = E D^{-1} and d (k,)
    """
    # E: (n,k) ; M: (n,n)
    # Compute M*E: (n,n)@(n,k) -> (n,k)
    ME = M @ E
    # d_i^2 = e_i^T M e_i = sum_j e_{j,i} * (M e_i)_{j}
    d2 = (E * ME).sum(dim=0)
    d = torch.sqrt(torch.clamp(d2, min=eps)).detach()
    Q = E / d.unsqueeze(0)
    return Q, d

def sol_apply(E, H_chol, M, b):
    """
    SOL in M-normalized basis:
      Q = E D^{-1}, D_ii = sqrt(e_i^T M e_i)
      x = Q H Q^T M b, with H = H_chol H_chol^T
    Shapes:
      E: (n,k), H_chol: (k,k), M: (n,n), b: (B,n) or (n,)
      Returns x: (B,n) or (n,)
    """
    Q, d = m_column_norms(E, M)

    # Build H = L L^T (SPD), we do not enforce it as a lower triangular.
    H = H_chol #  @ H_chol.t() + torch.eye(H_chol.shape[0], device=H_chol.device) * 1e-8

    if b.dim() == 1:
        y = (b.unsqueeze(0) @ Q).squeeze(0)  # (k,)
        z = y @ H                   # (k,)
        x = z @ Q.t()               # (n,)
        return x, Q
    else:
        y = b @ Q                  # (B,k)
        z = y @ H                   # (B,k)
        x = z @ Q.t()               # (B,n)
        return x, Q

def kkt_residual(E, A, M, b, x):
    """
    KKT residual in the M-normalized basis Q:
      r = Q^T (A x - M b)
    Returns r: (B,k) or (k,), and s: (k,) with s_i = q_i^T A q_i for normalization.
    """
    Q, _ = m_column_norms(E, M)
    if x.dim() == 1:
        r = Q.t() @ (A @ x - b)    # (k,)
    else:
        Ax = x @ A.t()                 # (B,n)
        r = (Ax - b) @ Q              # (B,k)
    # s_i = q_i^T A q_i (diag of Q^T A Q)
    AQ = A @ Q                         # (n,k)
    s = (Q * AQ).sum(dim=0)            # (k,)
    return r, s

def rayleigh_ritz_in_subspace(E_np, A_np, M_np, k):
    """
    Rayleigh–Ritz: solve (E^T A E) c = λ (E^T M E) c; return eigvals and ambient vectors E c normalized in M-norm.
    """
    K = E_np.T @ A_np @ E_np
    S = E_np.T @ M_np @ E_np
    S_chol = np.linalg.cholesky(S)
    S_inv = np.linalg.inv(S_chol)
    C = S_inv @ K @ S_inv.T
    w, U = np.linalg.eigh(C)  # ascending
    Cvecs = S_inv.T @ U
    V = E_np @ Cvecs
    # M-normalize and stabilize sign
    for j in range(k):
        norm = np.sqrt(V[:, j].T @ M_np @ V[:, j])
        if norm > 0:
            V[:, j] /= norm
        if V[0, j] < 0:
            V[:, j] *= -1
    return w, V

def true_eigs(A_np, M_np, k):
    """
    Compute k smallest generalized eigenpairs A v = λ M v using NumPy (eval only).
    """
    M_chol = np.linalg.cholesky(M_np)
    Minv = np.linalg.inv(M_chol)
    C = Minv @ A_np @ Minv.T
    w, U = np.linalg.eigh(C)
    idx = np.argsort(w)
    w = w[idx][:k]
    U = U[:, idx[:k]]
    V = np.linalg.solve(M_chol.T, U)  # v = M^{-1/2} u
    for j in range(k):
        norm = np.sqrt(V[:, j].T @ M_np @ V[:, j])
        if norm > 0:
            V[:, j] /= norm
        if V[0, j] < 0:
            V[:, j] *= -1
    return w, V

def main():
    parser = argparse.ArgumentParser(description="1D heat problem subspace demo (scale-invariant losses)")
    parser.add_argument("--n", type=int, default=128, help="number of grid points")
    parser.add_argument("--k", type=int, default=4, help="subspace dimension")
    parser.add_argument("--epochs", type=int, default=1000, help="training epochs")
    parser.add_argument("--batch", type=int, default=16, help="mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-2, help="initial learning rate")
    parser.add_argument("--delta", type=float, default=1e-3, help="shift for A = L + delta*M")
    parser.add_argument("--comp-weight", type=float, default=1.0, help="weight for compliance term")
    parser.add_argument("--kkt-weight", type=float, default=1.0, help="weight for KKT residual term")
    parser.add_argument("--plot-every", type=int, default=100, help="plot/print interval")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"], help="device")
    parser.add_argument("--savefig", type=str, default="", help="path to save final plot (optional)")
    parser.add_argument("--no-show", action="store_true", help="do not show interactive plot")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="gradient clip norm")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    # Build FEM operators
    x, L, M, w_lump, A = build_1d_fem(args.n, args.delta, args.k)
    x, L, M, A, w_lump = x.to(device), L.to(device), M.to(device), A.to(device), w_lump.to(device)
    vol = w_lump.sum().item()  # ≈ |Ω| = 1

    # Trainables: E (n×k) and H via its Cholesky param (k×k)
    E = nn.Parameter(torch.randn(args.n, args.k, device=device) / np.sqrt(args.n))
    H_chol = nn.Parameter(torch.eye(args.k, device=device) * 0.5 + 0.01 * torch.randn(args.k, args.k, device=device))
    params = [E, H_chol]

    optimizer = optim.Adam(params, lr=args.lr)
    # Exponential LR scheduler: final LR = 1e-2 × initial
    gamma = 0.01 ** (1.0 / max(args.epochs, 1))
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # Logging
    total_hist, comp_hist, kkt_hist, lr_hist, norm_hist = [], [], [], [], []

    # Precompute true eigenpairs (NumPy)
    A_np = A.detach().cpu().numpy()
    M_np = M.detach().cpu().numpy()
    true_vals, true_vecs = true_eigs(A_np, M_np, args.k)
    print("True smallest generalized eigenvalues (A v = λ M v):")
    print(true_vals)

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()

        # Sample batch of L2-isotropic loads, scaled for resolution invariance
        b = sample_b(args.batch, w_lump, device, scale_to_vol=True)  # (B,n)

        # SOL: x̂ = Q H Q^T M b (with Q = M-normalized columns of E)
        x_hat, Q = sol_apply(E, H_chol, M, b)           # (B,n)

        Ax_hat = x_hat @ A.t() # (B,n)


        # Compliance per unit volume: mean over batch of (b^T M x̂)/vol
        comp_batch = (b * x_hat).sum(dim=1) / vol
        comp = - comp_batch.mean()
        # comp = 1-torch.nn.functional.cosine_similarity(Mb, x_hat, dim=1).mean()

        # KKT residual: r = Q^T (A x̂ - M b), normalized by s_i = q_i^T A q_i, then mean over batch and k
        kkt = 0.
        #! Original version does not work
        # r, s = kkt_residual(E, A, M, b, x_hat)      # r: (B,k), s: (k,)
        # s_clamped = torch.clamp(s, min=1e-12)
        # kkt = (r.pow(2) / s_clamped.unsqueeze(0)).mean() / vol  # dimensionless, resolution-invariant
        #? This works, why?
        kkt = kkt + (0.5 * (Ax_hat * x_hat).sum(dim=1) - (b * x_hat).sum(dim=1)).mean() / vol

        # norm of E => non degenerating, i.e. | E.T M E - I |_F
        # <e_i, M e_j> = e_i^T M e_j.
        # wish dETME_ij/dE_i = 0 for all i > j.
        # a little bit wild: This encourage the earlier columns of E to explore more.
        # therefore we do not use the full matrix ETME, but only the lower triangular part
        ME = M @ E
        ortho = torch.tril(E.T.detach() @ ME / vol, diagonal=1)
        norm_E = (E * ME).sum(dim=0) / vol # we enable the grad for E now.
        # why tril: i > j is enabled. but grad to i is zero.
        norm = ortho.square().mean() + norm_E.mean()

        # Total loss: mean-based aggregations stabilize gradients across N and k
        loss = args.comp_weight * comp + args.kkt_weight * kkt + 1e-8 * norm
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(params, max_norm=args.grad_clip_norm)

        optimizer.step()
        scheduler.step()

        total_hist.append(loss.detach().cpu().item())
        comp_hist.append(comp.detach().cpu().item())
        kkt_hist.append(kkt.detach().cpu().item())
        lr_hist.append(optimizer.param_groups[0]["lr"])
        norm_hist.append(norm.detach().cpu().item())

        if epoch % args.plot_every == 0 or epoch == 1 or epoch == args.epochs:
            with torch.no_grad():
                E_np = E.detach().cpu().numpy()
                rr_vals, rr_vecs = rayleigh_ritz_in_subspace(E_np, A_np, M_np, args.k)
            print(f"Epoch {epoch:4d} | lr {optimizer.param_groups[0]['lr']:.3e} "
                  f"| total {np.mean(total_hist[-args.plot_every:]):.3e} "
                  f"| comp {np.mean(comp_hist[-args.plot_every:]):.3e} "
                  f"| kkt {np.mean(kkt_hist[-args.plot_every:]):.3e}"
                  f"| norm {np.mean(norm_hist[-args.plot_every:]):.3e}")
            print("  Ritz eigvals (learned subspace):", rr_vals)
            print("  True eigvals:                    ", true_vals)

    # Final evaluation and plots
    with torch.no_grad():
        E_np = E.detach().cpu().numpy()
        rr_vals, rr_vecs = rayleigh_ritz_in_subspace(E_np, A_np, M_np, args.k)

    fig, axs = plt.subplots(3, 1, figsize=(9, 10))
    # Plot loss curves
    axs[0].plot(total_hist, label="total")
    axs[0].plot(comp_hist, label="compliance")
    axs[0].plot(kkt_hist, label="KKT residual")
    axs[0].set_xlabel("epoch")
    axs[0].set_ylabel("loss")
    axs[0].set_title("Training losses (scale-invariant)")
    axs[0].legend()

    # Plot LR schedule
    axs[1].plot(lr_hist)
    axs[1].set_xlabel("epoch")
    axs[1].set_ylabel("learning rate")
    axs[1].set_title("Exponential LR schedule")

    # Plot first k learned vs true eigenfunctions
    ax = axs[2]
    x_np = x.detach().cpu().numpy()
    colors = plt.cm.tab10.colors
    for j in range(args.k):
        ax.plot(x_np, true_vecs[:, j], color=colors[j % 10], linestyle="-", label=f"true e{j+1}")
        ax.plot(x_np, rr_vecs[:, j],   color=colors[j % 10], linestyle="--", label=f"learned e{j+1}")

        print(f"True eigval {true_vals[j]:.3e} | Ritz eigval {rr_vals[j]:.3e} | CosSim {torch.nn.functional.cosine_similarity(torch.from_numpy(true_vecs[:, j]), torch.from_numpy(rr_vecs[:, j]), dim=0).item():.3e}")
    ax.set_title("True vs learned (Ritz) eigenfunctions")
    ax.set_xlabel("x")
    ax.set_ylabel("value")
    ax.legend(ncol=2)

    plt.tight_layout()
    if args.savefig:
        plt.savefig(args.savefig, dpi=150)
        print(f"Saved figure to {args.savefig}")
    if not args.no_show:
        plt.show()

if __name__ == "__main__":
    main()
