"""
This file is modified from 

MIT License

Copyright (c) 2020-2021 Nicholas Sharp and coauthors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch

def compute_correspondence(feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y, lambda_param=1e-3):
    """
    Computes the functional map correspondence matrix C given features from two shapes.

    Has no trainable parameters.
    """
    if feat_x.dim() == 2:
        feat_x, feat_y = feat_x.unsqueeze(0), feat_y.unsqueeze(0)
        evecs_trans_x, evecs_trans_y = evecs_trans_x.unsqueeze(0), evecs_trans_y.unsqueeze(0)
        evals_x, evals_y = evals_x.unsqueeze(0), evals_y.unsqueeze(0)

    F_hat = torch.bmm(evecs_trans_x, feat_x)
    G_hat = torch.bmm(evecs_trans_y, feat_y)
    A, B = F_hat, G_hat

    D = torch.repeat_interleave(evals_x.unsqueeze(1), repeats=evals_x.size(1), dim=1)
    D = (D - torch.repeat_interleave(evals_y.unsqueeze(2), repeats=evals_x.size(1), dim=2)) ** 2

    A_t = A.transpose(1, 2)
    A_A_t = torch.bmm(A, A_t)
    B_A_t = torch.bmm(B, A_t)

    C_i = []
    for i in range(evals_x.size(1)):
        D_i = torch.cat([torch.diag(D[bs, i, :].flatten()).unsqueeze(0) for bs in range(evals_x.size(0))], dim=0)
        # C = torch.bmm(torch.inverse(A_A_t + lambda_param * D_i), B_A_t[:, i, :].unsqueeze(1).transpose(1, 2))
        system_matrix = A_A_t + lambda_param * D_i
        rhs = B_A_t[:, i, :].unsqueeze(1).transpose(1, 2)
        C = torch.linalg.solve(system_matrix, rhs)
        C_i.append(C.transpose(1, 2))
    C = torch.cat(C_i, dim=1)

    return C

def compute_correspondence_batched(feat_x, feat_y, evals_x, evals_y, 
                                   evecs_trans_x, evecs_trans_y, lambda_param=1e-3):
    F_hat = torch.bmm(evecs_trans_x, feat_x)
    G_hat = torch.bmm(evecs_trans_y, feat_y)
    A, B = F_hat, G_hat

    D = torch.repeat_interleave(evals_x.unsqueeze(1), repeats=evals_x.size(1), dim=1)
    D = (D - torch.repeat_interleave(evals_y.unsqueeze(2), repeats=evals_x.size(1), dim=2)) ** 2

    A_t = A.transpose(1, 2)
    A_A_t = torch.bmm(A, A_t)
    B_A_t = torch.bmm(B, A_t)

    D_diag = torch.diag_embed(D)
    M = A_A_t.unsqueeze(1) + lambda_param * D_diag

    bsize = feat_x.size(0)
    k = evals_x.size(1)

    # Use Cholesky decomposition for numerical stability
    system_matrix_reshaped = M.reshape(bsize * k, k, k)
    L = torch.linalg.cholesky(system_matrix_reshaped)
    rhs_reshaped = B_A_t.reshape(bsize * k, k).unsqueeze(-1)
    C = torch.cholesky_solve(rhs_reshaped, L)
    C = C.squeeze(-1).reshape(bsize, k, k)

    return C

if __name__ == '__main__':
    torch.manual_seed(0)
    B = 3
    n = 128
    k = 32
    d = 16
    lambda_param = 1e-3

    feat_x = torch.randn(B, n, d)
    feat_y = torch.randn(B, n, d)
    evecs_trans_x = torch.randn(B, k, n)
    evecs_trans_y = torch.randn(B, k, n)
    evals_x = torch.rand(B, k) * 10.0 + 1.0
    evals_y = evals_x + torch.rand(B, k) * 0.5 + 0.1

    C_vec = compute_correspondence_batched(
        feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y, lambda_param=lambda_param
    )

    C_loop_list = []
    for b in range(B):
        C_b = compute_correspondence(
            feat_x[b], feat_y[b], evals_x[b], evals_y[b], evecs_trans_x[b], evecs_trans_y[b], lambda_param=lambda_param
        ).squeeze(0)
        C_loop_list.append(C_b)
    C_loop = torch.stack(C_loop_list, dim=0)

    diff = (C_loop - C_vec).abs()
    print('shape', C_vec.shape)
    print('allclose', torch.allclose(C_loop, C_vec, rtol=1e-5, atol=1e-6))
    print('max_abs_diff', diff.max().item())
    print('mean_abs_diff', diff.mean().item())
    
    C_loop = compute_correspondence(
        feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y, lambda_param=lambda_param
    )
    diff = (C_loop - C_vec).abs()
    print('shape', C_loop.shape)
    print('allclose', torch.allclose(C_loop, C_vec, rtol=1e-5, atol=1e-6))
    print('max_abs_diff', diff.max().item())
    print('mean_abs_diff', diff.mean().item())