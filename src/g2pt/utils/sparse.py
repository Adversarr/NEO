import functools
from typing import Any, Callable, TypeVar

import numpy as np
from scipy import sparse as sp
import torch
from torch import nn
from torch_geometric.nn import MessagePassing

from g2pt.utils.common import ensure_numpy

T = TypeVar("T")


def to_torch_sparse_csr(mat, dtype=torch.float32) -> torch.Tensor:
    """
    Convert matrix to torch sparse CSR tensor.
    """

    if isinstance(mat, np.ndarray):
        # Ensure dense ndarray -> torch CSR with correct dtype
        return torch.from_numpy(mat).to_sparse_csr().to(dtype=dtype)

    try:
        sparse = sp.csr_matrix(mat)
    except:
        print("Input matrix must be convertible to CSR matrix")
        raise

    indptr = sparse.indptr
    indices = sparse.indices
    data = sparse.data

    out = torch.sparse_csr_tensor(
        # Torch requires int64 indices
        crow_indices=torch.from_numpy(indptr.astype(np.int64)),
        col_indices=torch.from_numpy(indices.astype(np.int64)),
        values=torch.from_numpy(data).to(dtype=dtype),
        size=mat.shape,
    )
    return out.to(dtype=dtype)


def to_torch_sparse_coo(mat, dtype=torch.float32) -> torch.Tensor:
    """
    Convert matrix to torch sparse COO tensor.
    """

    if isinstance(mat, np.ndarray):
        return torch.from_numpy(mat).to_sparse_coo().to(dtype=dtype)

    try:
        sparse = sp.coo_matrix(mat)
    except:
        print("Input matrix must be convertible to COO matrix")
        raise

    row = sparse.row
    col = sparse.col
    data = sparse.data

    out = torch.sparse_coo_tensor(
        # Torch requires int64 indices
        indices=torch.from_numpy(np.vstack((row.astype(np.int64), col.astype(np.int64)))),
        values=torch.from_numpy(data).to(dtype=dtype),
        size=mat.shape,
    )
    return out.to(dtype=dtype)


def from_torch_sparse_csr(mat: torch.Tensor) -> sp.csr_matrix:
    """
    Convert torch sparse CSR tensor to scipy CSR matrix.
    """
    if not mat.is_sparse_csr:
        print("Input matrix must be torch sparse CSR tensor")
        raise ValueError

    # Move to CPU before converting to numpy
    crow_indices = mat.crow_indices().cpu().numpy()
    col_indices = mat.col_indices().cpu().numpy()
    values = mat.values().cpu().numpy()

    return sp.csr_matrix((values, col_indices, crow_indices))


def _ensure_contiguous(fn: Callable[..., T]) -> Callable[..., T]:
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        def maybe_to_contiguous(x):
            return x.contiguous() if isinstance(x, torch.Tensor) and (not x.is_sparse and not x.is_sparse_csr) else x

        args = [maybe_to_contiguous(arg) for arg in args]
        kwargs = {k: maybe_to_contiguous(v) for k, v in kwargs.items()}
        return fn(ctx, *args, **kwargs)

    return wrapper


class SymmSparseCSRMatmul(torch.autograd.Function):
    """
    Acceleration to Symmetric sparse CSR matrix-vector/matrix multiplication.
    Forward returns dense result; gradient flows only to the input vector.
    """

    @staticmethod
    @_ensure_contiguous
    def forward(ctx: torch.autograd.function.FunctionCtx, matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        assert not matrix.requires_grad, "Matrix must not require gradient"
        ctx.save_for_backward(matrix)
        return torch.sparse.mm(matrix, vector)

    @staticmethod
    @_ensure_contiguous
    def backward(ctx: torch.autograd.function.FunctionCtx, dmatmul: torch.Tensor) -> tuple[None, torch.Tensor]:
        (a,) = ctx.saved_tensors
        return None, torch.sparse.mm(a, dmatmul)

class GraphSpmv(MessagePassing):
    """Perform Sparse Matrix-Vector multiplication, in a message passing manner.

        Blocked Sparse matrix multiplication:
            $$ y_i = sum_j A_{i,j} * x_j $$
        where:
            - A is a sparse matrix, blocked, A_{i, j} is R^{dim X dim} matrix
            - x_j is a vector, R^{dim}
            - y_i is a vector, R^{dim}

    Attributes:
        transpose: Whether to transpose the matrix.
    """

    def __init__(self, use_transpose=False):
        flow = "target_to_source" if not use_transpose else "source_to_target"
        self.transpose = use_transpose
        super(GraphSpmv, self).__init__(aggr="add", flow=flow)

    def forward(self, X, edge_index, A):
        # (npoints, d)
        # (2, nnz)
        # (nnz, )
        out = self.propagate(edge_index, x=X, edge_attr=A.unsqueeze(-1))
        return out

    def message(self, x_i, x_j, edge_attr):  # type: ignore
        result = edge_attr * x_j  # (1, ) * (d, ) -> (d, )
        return result


def cpu_spmv(x, A_ind, A_val):
    """Sparse matrix-vector multiplication on CPU.

    Args:
        x: [B*N, nsol] input vector.
        A_ind: [2, nnz] indices of non-zero elements in A.
        A_val: [nnz] values of non-zero elements in A.

    Returns:
        y: [B*N, nsol] output vector.
    """
    x_np = ensure_numpy(x)
    A_ind = ensure_numpy(A_ind).reshape(2, -1)
    A_val = ensure_numpy(A_val).flatten()
    import scipy.sparse as sp

    coo = sp.coo_matrix((A_val, (A_ind[0], A_ind[1])), shape=(x_np.shape[0], x_np.shape[0]))

    return torch.from_numpy(coo @ x_np).to(dtype=x.dtype, device=x.device)


class SymmetricSpmm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, A_ind, A_val):
        if A_val.dtype != torch.float32:
            A_val = A_val.float()
        coo = torch.sparse_coo_tensor(
            indices=A_ind,
            values=A_val,
            size=(X.shape[0], X.shape[0]),
        ).coalesce()
        ctx.save_for_backward(coo)
        return torch.sparse.mm(coo, X)

    @staticmethod
    def backward(ctx, dY):
        (coo,) = ctx.saved_tensors
        return torch.sparse.mm(coo.t(), dY), None, None


class NativeSpmv(nn.Module):
    def __init__(self, use_transpose=False):
        super().__init__()
        self.use_transpose = use_transpose

    def forward(self, x, A_ind, A_val):
        return SymmetricSpmm.apply(x, A_ind, A_val)


if __name__ == "__main__":
    N = 1024
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    mat = sp.random(N, N, density=0.003, format="csr")
    mat = mat + mat.T
    m = to_torch_sparse_csr(mat)
    m = m.to(dev)
    rhs = np.random.randn(N)
    b = torch.from_numpy(rhs).float().to(dev)

    gt = torch.from_numpy(mat @ rhs).float().to(dev)
    pred = m @ b
    print("Test to_torch_sparse_csr:", (torch.norm(pred - gt) < 1e-5).item())

    b = b.requires_grad_(True)

    with torch.enable_grad():
        mb = SymmSparseCSRMatmul.apply(m, b)
        mb.sum().backward()
        print("Test SymmSparseCSRMatmul-vector:", (torch.norm(b.grad - m @ torch.ones_like(mb)) < 1e-5).item())

    with torch.enable_grad():
        c = torch.randn(N, 4).to(dev).requires_grad_(True)
        mc = SymmSparseCSRMatmul.apply(m, c)
        mc.sum().backward()
        print("Test SymmSparseCSRMatmul-matrix:", (torch.norm(c.grad - m @ torch.ones_like(mc)) < 1e-5).item())

    import timeit

    c = torch.randn(N, 4).to(dev).requires_grad_(True)

    def do_native_bwd():
        mc = m @ c
        mc.sum().backward()

    def do_sparse_bwd():
        mc = SymmSparseCSRMatmul.apply(m, c)
        mc.sum().backward()

    print("Native backward time:", timeit.timeit(do_native_bwd, number=1000) / 1000)
    print("Sparse backward time:", timeit.timeit(do_sparse_bwd, number=1000) / 1000)

    # Check perm works
    m_perm = mat[: N // 2, : N // 2]
    print(m_perm)
