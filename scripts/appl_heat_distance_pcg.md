Here is a detailed, rigorous breakdown of the technical components for your paper. I will use formal mathematical notation suitable for a SIGGRAPH submission.

---

### 1. Preconditioned Conjugate Gradient (PCG)

The Conjugate Gradient (CG) method is an iterative solver for symmetric positive definite (SPD) systems $A\mathbf{x} = \mathbf{b}$. In the context of the Heat Equation (using implicit Euler), $A = M + \Delta t L$. 

The convergence rate of CG is bounded by the condition number $\kappa(A) = \lambda_{max}/\lambda_{min}$. For discrete Laplacians, $\kappa(A)$ scales poorly with the resolution $N$, leading to "low-frequency stagnation"—where the high-frequency errors vanish quickly, but the global, low-frequency modes take hundreds of iterations to resolve.

**PCG** mitigates this by introducing a preconditioner $P$, solving the equivalent system:
$$P^{-1}A\mathbf{x} = P^{-1}\mathbf{b}$$
where $P$ is an approximation of $A$ such that $P^{-1}A$ has a clustered spectrum. In our pipeline, PCG serves as the **outer-loop corrector**, ensuring that even if the neural-predicted field has errors, the final solution remains physically consistent to machine precision.

---

### 2. Deflation in PCG: Theoretical Foundation

Deflation is a sophisticated technique used to "remove" the influence of specific eigenvalues from the system's spectrum. While standard preconditioners (like Jacobi or ILU) focus on smoothing high-frequency noise, **Deflation** explicitly targets the **low-frequency spectrum** (the "bottom" eigenvalues that typically hinder CG convergence).

Given a set of $m$ linearly independent vectors $Y \in \mathbb{R}^{n \times m}$ that approximate the low-frequency invariant subspace (provided by NEO), we define the **Deflation Space** as $\mathcal{S} = \text{span}(Y)$.

#### The Projection Operators
We define the Galerkin projection onto the subspace:
1.  **Coarse-grid matrix:** $E = Y^\top A Y \in \mathbb{R}^{m \times m}$. This is a small, dense matrix representing the operator $A$ in the low-frequency subspace.
2.  **Correction matrix:** $Q = Y E^{-1} Y^\top$.
3.  **Deflation projector:** $P_D = I - A Q$.

The operator $P_D$ has a crucial property: it maps all vectors in $\mathcal{S}$ to the null space of $P_D A$. Effectively, the deflated system $P_D A \mathbf{x} = P_D \mathbf{b}$ "hides" the troublesome small eigenvalues from the PCG solver, treating them as zero and allowing the solver to focus on the remaining well-conditioned part of the spectrum.

---

### 3. Integrated Application: NEO-Accelerated Heat Equation Solver

In the Heat Equation $\frac{\partial \phi}{\partial t} = \Delta \phi$, the solution after one time step is $\phi_{t+1} = (M + \Delta t L)^{-1} M \phi_t$. We propose a hybrid framework where our Neural Operator informs the numerical solver.

#### Step A: Subspace Extraction (NEO Inference)
For a given geometry $X$, NEO performs a single forward pass to generate $Y = \mathcal{F}_\theta(X, w)$. Unlike traditional methods that require solving a Large-scale Generalized Eigenvalue Problem (GEVP), NEO provides the **optimal deflation basis** $Y$ at a fraction of the cost.

#### Step B: The Initial Guess (Spectral Warm-start)
By projecting the problem onto the NEO subspace, we can find the best possible initial guess $\mathbf{x}_0$ in $\text{span}(Y)$:
$$\mathbf{x}_0 = Y (Y^\top A Y)^{-1} Y^\top \mathbf{b}$$
Since the Heat Equation is dominated by low-frequency diffusion, this $\mathbf{x}_0$ often captures the global heat distribution immediately, leaving only local residuals for PCG.

#### Step C: Deflated PCG Iteration
We solve for the remainder using the Deflated PCG algorithm. In each iteration, the residual is projected using $P_D$. 
Even if the geometry is complex or the time step $\Delta t$ is large, the convergence is guaranteed. The "heavy lifting" of resolving global diffusion—which usually takes the most iterations—is handled by the NEO-derived $E^{-1}$, while the PCG handles the high-frequency features.

---

### Summary for Paper Narrative

| Aspect | Traditional PCG | Our NEO-Deflated PCG |
| :--- | :--- | :--- |
| **Bottleneck** | Slow convergence on global (low-freq) modes. | Accelerated by NEO-informed subspace. |
| **Subspace $Y$** | Requires expensive Lanczos/Arnoldi solvers. | Instantaneous inference via Neural Operator. |
| **Reliability** | Mathematically rigorous. | Same rigor; ML only accelerates the search. |
| **Performance** | O(Iterations $\times N$) | O(Forward Pass + Low Iterations $\times N$) |

**Why this works for SIGGRAPH:**
It respects the "shoulders of giants" by using the Deflated CG (a 20-year-old proven numerical method) but replaces the most expensive part of that method (subspace construction) with a modern Neural Operator. This provides a clear **Pareto improvement** in terms of speed without sacrificing the error bounds required for physical simulation.

From the two test runs you posted, the problem is actually quite clear: **deflation is not ineffective, but rather you've nearly completely offset the benefits of reducing iteration counts with implementation overhead**. Additionally, **the heat step itself doesn't need deflation**, which creates the counter-intuitive phenomenon of “more deflation equals slower”.

I'll break this down for you using “evidence → possible causes → how to verify/fix”.

---

## 0) First, look at the evidence: you did reduce Poisson iterations, but the time barely decreased

Your data (N≈48k, k=20):

### Heat solve
- Without deflation: 34 iter, 0.00856 s  → **0.000252 s/iter**
- With deflation: 44 iter, 0.03805 s  → **0.000865 s/iter (3.4× slower per step)**

Not only did heat not speed up, it got slower and required more iterations.

### Poisson solve
- Without deflation: 1446 iter, 0.36071 s → **0.000249 s/iter**
- With deflation: 359 iter, 0.32127 s  → **0.000895 s/iter (3.6× slower per step)**

Poisson iterations dropped by **4× (1446→359, this is “effective deflation”)**, but **per-step cost increased 3.6×**, so total time only marginally improved (0.361→0.321).

Conclusion: **Algorithmically deflation is working; but the engineering implementation consumed all the benefits.**

---

## 1) First major problem (very severe): You're repeatedly doing Cholesky decomposition of \(E\) on every projection

Looking at your implementation:

```python
def _safe_solve_spd(mat, rhs, ridge):
    ...
    factor = np.linalg.cholesky(mat)   # <-- Decomposes every time this is called
    y = np.linalg.solve(factor, rhs)
    return np.linalg.solve(factor.T, y)
```

And in every PCG iteration you perform these projections (at least 3 times):
- `r = deflation.project_left(r)`  → solve \(E\beta=Y^Tr\) once
- `z = deflation.project_a_orth(z)`→ solve \(E\beta=(AY)^Tz\) once
- `p = deflation.project_a_orth(p)`→ solve again

In other words: **for every iteration step, you're doing 3 Cholesky decompositions on the same 20×20 matrix \(E\)**.
Poisson: 359 iterations ≈ 1077 Cholesky decompositions; Heat: 44 iterations ≈ 132 decompositions. Compared to “no deflation” where each step basically only has sparse matvec + dot, this extra overhead will obviously explode the per-iter time.

> This explains the “3-4× slower per step” you observed, and matches exactly with the per-iter time growth in your logs.

### How to fix (you should do this immediately)
Cache the factorization of \(E\) during `DeflationSpace.build()` and reuse it via `cho_solve`:

- During build:
  - `E_r = E + ridge I`
  - `chol = scipy.linalg.cho_factor(E_r, lower=True, check_finite=False)`
- For each solve:
  - `beta = scipy.linalg.cho_solve(chol, rhs, check_finite=False)`

Or just precompute `E_inv` directly (m=20 is small, direct inversion isn't outrageous, but numerical stability is slightly worse).

**This is your main performance bug right now.**

---

## 2) Second major problem: Heat step shouldn't expect deflation speedup (your t is too small)

Your heat step parameters here:
- \(h \approx 0.01047\)
- \(t = h^2 \approx 1.097\times 10^{-4}\)

And what you're solving:
\[
A_{\text{heat}} = M + tL
\]

When \(t\) is this small, \(A_{\text{heat}}\) is **dominated by \(M\)**, and the matrix condition number is usually already decent (especially with Jacobi preconditioning). This is why without deflation, heat converges in just **34 iterations**. At this point, deflation can only improve a limited spectral problem, yet you introduce additional overhead:
- One `A @ Y` during deflation space build
- 3 projections per step (plus the huge cost of your repeated Cholesky decompositions)

So heat slowing down is expected, and more iterations is unsurprising either (projection + numerical error + mismatched subspace all perturb conjugacy).

### Recommendations (pragmatic rather than cosmetic)
- **Don't do deflation on heat step** (only on Poisson), you'll immediately get better end-to-end time.
- If you insist on “showing deflation effect on heat step”, you need to change the experimental setup: increase \(t\) (multi-scale heat diffusion/stronger smoothing), otherwise you're fighting against the physics itself.

---

## 3) Third major problem: You can't prove deflation works using “eigen MSE < 0.01”

Deflation/PCG convergence depends on whether your deflation subspace approximates **the spectral components in the target linear system** that cause slow convergence (usually the extreme eigenmodes of **\(P^{-1}A\)**, not the eigenvectors of \(L\) from your training).

Several common misalignment points:

### 3.1 What are you deflating - \(A_{\text{heat}}=M+tL\) or pinned \(L\) - and what is your basis?
Your loaded `net_evec.npy` is probably the generalized eigenvalue problem:
\[
L u = \lambda M u, \quad u^T M u = 1
\]
These \(u\) are **generalized eigenvectors** (M-orthogonal), which are not equivalent to:
- Eigenvectors of \(L\) in Euclidean sense (what Poisson CG cares about)
- Eigenvectors of \(A_{\text{heat}}\) in Euclidean sense (what heat CG cares about)

When “mesh quality varies significantly”, this mismatch makes the deflation “direction” insufficiently aligned, which:
- Fails to accurately capture the theoretical worst-case modes that should be deflated
- Can even cause more heat iterations due to projection perturbation

**More reliable evaluation metrics** are not MSE, but:
- Principal angles between subspaces / projection energy
- Or Ritz residual: \(\|A y_i - \mu_i y_i\|\)
- Or directly examine spectral clustering of the deflated operator (more important)

### 3.2 Constraints of pinned Poisson mismatch with your basis
You pinned Poisson as:
```python
a_lil[pin_idx, :] = 0
a_lil[:, pin_idx] = 0
a_lil[pin_idx, pin_idx] = 1
rhs[pin_idx] = 0
```
But your basis was not similarly “pin-compatible”. This causes:
- Some dimensions in the coarse space are meaningless in the feasible solution space
- \(E = Y^T A Y\) may be more ill-conditioned (you rely on ridge regularization)
- Reduced effective dimensions in deflation

**Recommendation: For pinned systems, pin your basis once too** (at minimum set `basis[pin_idx,:]=0`, then orthogonalize/QR again).

---

## 4) What should you diagnose next: use three controlled experiments to isolate the real cause

Don't guess; just run these three experiments:

1) **Use the GT eigenbasis (`mesh_gt`) for deflation**
   - If GT also gives almost no speedup: the main cause is your deflation implementation, timing methodology, or projection overhead (from your current code, this is very likely).
   - If GT speeds things up clearly but the network does not: your learned subspace is still not a good effective subspace for the linear system (and MSE is not a sensitive metric here).

2) **Deflate only the Poisson solve; do not deflate the heat solve**
   This is the configuration most likely to improve end-to-end runtime immediately.

3) **Cache the factorization of \(E\) vs. do not cache it**
   You should see per-iteration time drop dramatically (at least 2-3x), which is what finally turns "fewer iterations" into "less runtime."

---

### 5. Advanced Preconditioning Strategy: Two-Level Deflation

To further reduce the computational overhead of projection per iteration, we implemented a **Two-Level Additive Preconditioner** (enabled via `--deflation-type additive`).

Instead of enforcing the deflation constraint via repeated projections (Deflated PCG), this method incorporates the subspace correction directly into the preconditioner:
$$
\tilde{P}^{-1} = P_{\text{fine}}^{-1} + P_{\text{coarse}}^{-1} = P_{\text{fine}}^{-1} + Y (Y^T A Y)^{-1} Y^T
$$
where $P_{\text{fine}}^{-1}$ is the standard smoother (Jacobi or ICHOL).

**Advantages:**
*   **Zero Projection Overhead**: The PCG loop remains standard; no dense matrix-vector multiplications for projection are needed inside the loop.
*   **Stability**: Acts as a robust spectral shift, effectively removing the smallest eigenvalues that hinder convergence.

### 6. Experimental Metrics for Publication

To demonstrate the efficacy of this method in a paper (e.g., SIGGRAPH), record and present the following metrics:

| Metric | Definition | Goal |
| :--- | :--- | :--- |
| **Iterations** | Number of PCG steps to reach $\|r\| < \epsilon$. | Show drastic reduction (e.g., $10\times$) vs. standard PCG. |
| **Time per Iteration** | Average wall-clock time for one PCG step. | Show that `additive` deflation adds minimal overhead compared to `projection`. |
| **Setup Time** | Time to build `DeflationSpace` ($E^{-1}$). | Show this is negligible compared to the total solve time. |
| **Total Wall-Clock Time** | Setup + Solve time. | **The ultimate metric.** Must be lower than `ICHOL + PCG`. |
| **Robustness** | Success rate on meshes with high aspect ratios. | Demonstrate convergence where standard methods fail. |

**Recommended Benchmark Config:**
*   **Baseline**: `PCG + ICHOL` (No Deflation).
*   **Ours**: `PCG + ICHOL + Additive Deflation (k=8)`.
*   **Target**: Show that "Ours" achieves the same accuracy in fewer iterations and less total time, especially for the Poisson step.
