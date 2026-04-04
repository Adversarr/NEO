import torch
# torch.set_printoptions(precision=6, suppress=True)

def proj_loss_1(Q, V):
    # Corresponds to the first ProjectionLoss: p=2, dim=1, reduction='mean'
    Qt = Q.transpose(-2, -1)
    QQtV = torch.bmm(Q, torch.bmm(Qt, V))
    res = V - QQtV
    # L2 norm per column, then sum across columns
    per_col = torch.norm(res, dim=1)  # [B, K]: L2 norm per column
    loss_b = per_col.sum(dim=-1)      # [B]: sum across columns
    return loss_b.mean()

def proj_loss_2(Q, V, upper_mode='min'):
    # Corresponds to the forward body of the second ProjectionLoss
    # upper_mode='min' -> upper = 2 * min(k, K)
    # upper_mode='sum' -> upper = k + K (equivalent to projection matrix distance ||P_Q - P_V||_F)
    k = Q.shape[-1]
    K = V.shape[-1]
    if upper_mode == 'min':
        upper = 2 * min(k, K)
    else:
        upper = k + K
    utv = torch.bmm(Q.transpose(-2, -1), V)       # [B, k, K]
    err = upper - 2 * torch.sum(utv ** 2, dim=(-1, -2))  # [B]
    return torch.sqrt(err.clamp_min(0) + 1e-12).mean()

# Construct example
B, d, k, K = 1, 3, 2, 2
e1 = torch.tensor([[1.,0.,0.]]).T
e2 = torch.tensor([[0.,1.,0.]]).T
e3 = torch.tensor([[0.,0.,1.]]).T

theta = torch.tensor(30.0 * 3.1415926535 / 180.0)
v1 = torch.cos(theta) * e1 + torch.sin(theta) * e3  # cosθ e1 + sinθ e3

Q = torch.stack([e1.squeeze(), e2.squeeze()], dim=1).unsqueeze(0)  # [1,3,2]
V1 = torch.stack([v1.squeeze(), e2.squeeze()], dim=1).unsqueeze(0) # [1,3,2]

# Rotate 45° within V1's subspace to obtain another basis V2 spanning the same space
R = (1.0 / (2.0 ** 0.5)) * torch.tensor([[1., 1.], [1., -1.]])
V2 = torch.bmm(V1, R.unsqueeze(0))  # [1,3,2]

# Another example with the exact same subspace: rotate 45° within Q's subspace
R_Q = torch.tensor([[0.707107, -0.707107], [0.707107, 0.707107]])
V_same = torch.bmm(Q, R_Q.unsqueeze(0))  # same subspace as Q

# Compute and print
print("Results for theta=30°, sinθ=0.5:")
print("1) Different subspaces, using V1:")
print("   Loss1 =", proj_loss_1(Q, V1).item())            # expected ~ 0.5
print("   Loss2(min) =", proj_loss_2(Q, V1, 'min').item())# expected ~ 0.7071
print("   Loss2(sum) =", proj_loss_2(Q, V1, 'sum').item())# expected ~ 0.7071 (same when k=K)

print("\n2) Same subspace as 1 (same V span), but different basis V2:")
print("   Loss1 =", proj_loss_1(Q, V2).item())            # expected ~ 0.7071 (increases)
print("   Loss2(min) =", proj_loss_2(Q, V2, 'min').item())# expected ~ 0.7071 (unchanged)
print("   Loss2(sum) =", proj_loss_2(Q, V2, 'sum').item())

print("\n3) Exactly the same subspace (V_same spans same space as Q):")
print("   Loss1 =", proj_loss_1(Q, V_same).item())        # expected 0
print("   Loss2(min) =", proj_loss_2(Q, V_same, 'min').item())# expected 0
print("   Loss2(sum) =", proj_loss_2(Q, V_same, 'sum').item()) # expected 0
