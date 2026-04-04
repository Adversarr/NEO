import time
import torch
from torch import nn

# Optional Triton support. If unavailable or on non-CUDA, falls back to PyTorch path.
try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
    _TRITON_AVAILABLE = True # TODO: Enable triton on CUDA
except Exception:  # pragma: no cover
    _TRITON_AVAILABLE = False

def sinusoidal_frequencies(dim: int, min_log_freq: float = -5, max_log_freq: float = 3) -> torch.Tensor:
    """
    Generate sinusoidal frequencies for positional encoding.

    Args:
        dim (int): The dimension of the positional encoding.
        min_log_freq (float): The lower bound of the log frequency. Default is -5.
        max_log_freq (float): The upper bound of the log frequency. Default is 3.

    Returns:
        torch.Tensor: A tensor containing the sinusoidal frequencies.
    """
    half = dim // 2
    freqs = torch.pi * torch.exp2(torch.linspace(min_log_freq, max_log_freq, half, dtype=torch.float32))
    return freqs

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half the hidden dimensions of the input tensor.
    
    This function splits the input tensor's last dimension into two halves,
    negates the second half, and concatenates them in reverse order.
    This operation is a key component of rotary position embeddings.
    
    Args:
        x (torch.Tensor): Input tensor with shape (..., dim) where dim is even.
                         The last dimension will be split into two halves.
    
    Returns:
        torch.Tensor: Tensor with the same shape as input where the second half
                     of the last dimension is negated and swapped with the first half.
    
    Example:
        >>> x = torch.randn(2, 3, 8)  # shape (batch, seq, dim)
        >>> result = rotate_half(x)
        >>> result.shape  # (2, 3, 8) - same shape as input
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies Rotary Position Embedding to the query tensor.

    Args:
        q (torch.Tensor): Query tensor of shape (b, h, s, d) or (b, s, d). The last dim must be even.
        cos (torch.Tensor): Cosine values broadcastable to `q` (expected shape (b, 1, s, d) or (b, h, s, d)).
        sin (torch.Tensor): Sine values broadcastable to `q` (expected shape (b, 1, s, d) or (b, h, s, d)).

    Returns:
        torch.Tensor: Rotated query tensor with the same shape as `q`.
    """
    # Always go through a custom autograd path so gradients for cos/sin are produced.
    # The autograd function chooses Triton or eager depending on availability.
    
    # Check if cos/sin have head dimension (shape[1] > 1)
    # Triton only supports cos/sin without head dimension (shape[1] == 1 or no head dimension)
    cosine_has_head = cos.ndim == 4 and cos.shape[1] > 1
    sine_has_head = sin.ndim == 4 and sin.shape[1] > 1
    
    # Use Triton only if available and cos/sin don't have head dimension
    if _TRITON_AVAILABLE and not cosine_has_head and not sine_has_head:
        return _RoPE1DTritonFn.apply(q, cos, sin) # type: ignore
    else:
        # Use fallback when cos/sin have head dimension or Triton is unavailable
        return rope_apply_eager_fallback(q, cos, sin)


# =========================
# Triton kernels and helpers
# =========================

if _TRITON_AVAILABLE:
    @triton.jit
    def _triton_rope_q_forward(
        q_ptr,
        q_row_stride,
        cos_ptr,
        cos_row_stride,
        sin_ptr,
        sin_row_stride,
        sl,
        bs: tl.constexpr,
        cos_bs: tl.constexpr,
        n_qh: tl.constexpr,
        hd: tl.constexpr,
        pad_n_qh: tl.constexpr,
        pad_hd: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        # q: (bsz, seq, n_head, hd)
        pid = tl.program_id(0).to(tl.int64)

        q_ptr = q_ptr + pid * q_row_stride

        batch_idx = pid // sl
        cos_row_idx = pid % sl

        # cos/sin: (cos_bs, sl, hd)
        cos_ptr = cos_ptr + tl.where(
            cos_bs == 1,
            cos_row_idx * cos_row_stride,
            batch_idx * (sl * cos_row_stride) + cos_row_idx * cos_row_stride,
        )
        sin_ptr = sin_ptr + tl.where(
            cos_bs == 1,
            cos_row_idx * sin_row_stride,
            batch_idx * (sl * sin_row_stride) + cos_row_idx * sin_row_stride,
        )

        half_hd = hd // 2

        offs_half = tl.arange(0, pad_hd // 2)
        mask_half1 = offs_half < half_hd

        # Left half: cos1/sin1
        cos1 = tl.load(cos_ptr + offs_half, mask=mask_half1, other=0.0)
        sin1 = tl.load(sin_ptr + offs_half, mask=mask_half1, other=0.0)

        # Right half: cos2/sin2
        offs_half2 = offs_half + half_hd
        mask_half2 = offs_half2 < hd
        cos2 = tl.load(cos_ptr + offs_half2, mask=mask_half2, other=0.0)
        sin2 = tl.load(sin_ptr + offs_half2, mask=mask_half2, other=0.0)

        # First and second half offsets for q
        head_ids = tl.arange(0, pad_n_qh)[:, None]
        dim_ids = tl.arange(0, pad_hd // 2)[None, :]
        first_half_q_offsets = head_ids * hd + dim_ids
        second_half_q_offsets = first_half_q_offsets + half_hd

        head_mask = head_ids < n_qh
        dim_mask = dim_ids < half_hd
        q_mask = head_mask & dim_mask

        q1 = tl.load(q_ptr + first_half_q_offsets, mask=q_mask, other=0.0).to(sin1.dtype)
        q2 = tl.load(q_ptr + second_half_q_offsets, mask=q_mask, other=0.0).to(sin1.dtype)

        # y1 = q1*c1 - q2*s1
        # y2 = q2*c2 + q1*s2
        new_q1 = q1 * cos1 - q2 * sin1
        new_q2 = q2 * cos2 + q1 * sin2

        tl.store(q_ptr + first_half_q_offsets, new_q1, mask=q_mask)
        tl.store(q_ptr + second_half_q_offsets, new_q2, mask=q_mask)

    @triton.jit
    def _triton_rope_q_backward(
        dy_ptr,
        dy_row_stride,
        cos_ptr,
        cos_row_stride,
        sin_ptr,
        sin_row_stride,
        q_ptr,
        q_row_stride,
        dcos_ptr,
        dcos_row_stride,
        dsin_ptr,
        dsin_row_stride,
        sl,
        bs: tl.constexpr,
        cos_bs: tl.constexpr,
        n_qh: tl.constexpr,
        hd: tl.constexpr,
        pad_n_qh: tl.constexpr,
        pad_hd: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)

        dy_ptr = dy_ptr + pid * dy_row_stride
        q_ptr = q_ptr + pid * q_row_stride

        batch_idx = pid // sl
        cos_row_idx = pid % sl

        cos_ptr = cos_ptr + tl.where(
            cos_bs == 1,
            cos_row_idx * cos_row_stride,
            batch_idx * (sl * cos_row_stride) + cos_row_idx * cos_row_stride,
        )
        sin_ptr = sin_ptr + tl.where(
            cos_bs == 1,
            cos_row_idx * sin_row_stride,
            batch_idx * (sl * sin_row_stride) + cos_row_idx * sin_row_stride,
        )

        dcos_ptr = dcos_ptr + tl.where(
            cos_bs == 1,
            cos_row_idx * dcos_row_stride,
            batch_idx * (sl * dcos_row_stride) + cos_row_idx * dcos_row_stride,
        )
        dsin_ptr = dsin_ptr + tl.where(
            cos_bs == 1,
            cos_row_idx * dsin_row_stride,
            batch_idx * (sl * dsin_row_stride) + cos_row_idx * dsin_row_stride,
        )

        half_hd = hd // 2
        offs_half = tl.arange(0, pad_hd // 2)

        mask_half1 = offs_half < half_hd
        offs_half2 = offs_half + half_hd
        mask_half2 = offs_half2 < hd

        cos1 = tl.load(cos_ptr + offs_half,  mask=mask_half1, other=0.0)
        sin1 = tl.load(sin_ptr + offs_half,  mask=mask_half1, other=0.0)
        cos2 = tl.load(cos_ptr + offs_half2, mask=mask_half2, other=0.0)
        sin2 = tl.load(sin_ptr + offs_half2, mask=mask_half2, other=0.0)

        head_ids = tl.arange(0, pad_n_qh)[:, None]
        dim_ids = tl.arange(0, pad_hd // 2)[None, :]
        offsets1 = head_ids * hd + dim_ids
        offsets2 = offsets1 + half_hd

        head_mask = head_ids < n_qh
        dim_mask = dim_ids < half_hd
        mask_h = head_mask & dim_mask

        dy1 = tl.load(dy_ptr + offsets1, mask=mask_h, other=0.0).to(sin1.dtype)
        dy2 = tl.load(dy_ptr + offsets2, mask=mask_h, other=0.0).to(sin1.dtype)
        q1  = tl.load(q_ptr  + offsets1, mask=mask_h, other=0.0).to(sin1.dtype)
        q2  = tl.load(q_ptr  + offsets2, mask=mask_h, other=0.0).to(sin1.dtype)

        # Correct dq:
        # dq1 = dy1*c1 + dy2*s2
        # dq2 = dy2*c2 - dy1*s1
        dq1 = dy1 * cos1 + dy2 * sin2
        dq2 = dy2 * cos2 - dy1 * sin1

        tl.store(dy_ptr + offsets1, dq1, mask=mask_h)
        tl.store(dy_ptr + offsets2, dq2, mask=mask_h)

        # dcos/dsin: sum over head dimension (axis=0)
        dcos1 = tl.sum(dy1 * q1,     axis=0)
        dcos2 = tl.sum(dy2 * q2,     axis=0)
        dsin1 = tl.sum((-dy1) * q2,  axis=0)
        dsin2 = tl.sum(dy2 * q1,     axis=0)

        tl.store(dcos_ptr + offs_half,  dcos1, mask=mask_half1)
        tl.store(dcos_ptr + offs_half2, dcos2, mask=mask_half2)
        tl.store(dsin_ptr + offs_half,  dsin1, mask=mask_half1)
        tl.store(dsin_ptr + offs_half2, dsin2, mask=mask_half2)


    def _rope_apply_triton(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # Normalize layout to (b, h, s, d)
        has_head = q.dim() == 4
        if not has_head:
            q = q.unsqueeze(1)

        # Physical layout: (b, s, h, d)
        q_phys = q.transpose(1, 2).contiguous()
        bsz, seq_len, n_q_head, head_dim = q_phys.shape

        # cos/sin: (b, 1, s, d) -> (b, s, d)
        cos_full = cos.squeeze(1).contiguous()
        sin_full = sin.squeeze(1).contiguous()
        assert cos_full.shape == (bsz, seq_len, head_dim)
        assert sin_full.shape == (bsz, seq_len, head_dim)

        pad_hd = triton.next_power_of_2(head_dim)
        pad_n_qh = triton.next_power_of_2(n_q_head)
        BLOCK_SIZE = pad_n_qh

        cos_bs = cos_full.shape[0]   # 1 or bsz
        n_row = bsz * seq_len

        _triton_rope_q_forward[(n_row,)](
            q_phys,
            q_phys.stride(1),
            cos_full,
            cos_full.stride(-2),
            sin_full,
            sin_full.stride(-2),
            seq_len,
            bsz,
            cos_bs,
            n_q_head,
            head_dim,
            pad_n_qh,
            pad_hd,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        out = q_phys.transpose(1, 2)
        if not has_head:
            out = out.squeeze(1)
        return out

    def _rope_backward_triton(
        dy: torch.Tensor, q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        had_head = q.dim() == 4
        if not had_head:
            q = q.unsqueeze(1)
            dy = dy.unsqueeze(1)

        q_phys = q.transpose(1, 2).contiguous()
        dy_phys = dy.transpose(1, 2).contiguous()
        bsz, seq_len, n_q_head, head_dim = q_phys.shape

        cos_full = cos.squeeze(1).contiguous()  # (b, s, d)
        sin_full = sin.squeeze(1).contiguous()
        assert cos_full.shape == (bsz, seq_len, head_dim)
        assert sin_full.shape == (bsz, seq_len, head_dim)

        pad_hd = triton.next_power_of_2(head_dim)
        pad_n_qh = triton.next_power_of_2(n_q_head)
        BLOCK_SIZE = pad_n_qh

        cos_bs = cos_full.shape[0]
        n_row = bsz * seq_len

        dcos_full = torch.zeros_like(cos_full)
        dsin_full = torch.zeros_like(sin_full)

        _triton_rope_q_backward[(n_row,)](
            dy_phys,
            dy_phys.stride(1),
            cos_full,
            cos_full.stride(-2),
            sin_full,
            sin_full.stride(-2),
            q_phys,
            q_phys.stride(1),
            dcos_full,
            dcos_full.stride(-2),
            dsin_full,
            dsin_full.stride(-2),
            seq_len,
            bsz,
            cos_bs,
            n_q_head,
            head_dim,
            pad_n_qh,
            pad_hd,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # dy_phys now contains dq
        dq = dy_phys.transpose(1, 2)
        if not had_head:
            dq = dq.squeeze(1)

        # Restore to (b,1,s,d) to match the eager backward interface
        dcos = dcos_full.unsqueeze(1)
        dsin = dsin_full.unsqueeze(1)
        return dq, dcos, dsin




class _RoPE1DTritonFn(torch.autograd.Function):
    """
    Triton-backed autograd for applying RoPE to a single tensor `q`, while returning
    gradients w.r.t. `cos` and `sin` so upstream parameters (e.g., positions `x`) get gradients.

    Forward path uses Triton if available. Backward path computes:
      dq = dy * cos + rotate_half(dy) * sin
      dcos = dy * q
      dsin = dy * rotate_half(q)

    When Triton is available, dq is computed via Triton; dcos/dsin are aggregated across heads
    inside the Triton kernel and expanded to full last-dim length.
    """

    @staticmethod
    def forward(ctx, q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(q, cos, sin)
        if _TRITON_AVAILABLE and q.is_cuda:
            return _rope_apply_triton(q, cos, sin)
        else:
            assert False, "RoPE1DTritonFn only supports CUDA tensors"
        # # Eager fallback via helper
        # return rope_apply_eager_fallback(q, cos, sin)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        q, cos, sin = ctx.saved_tensors
        if _TRITON_AVAILABLE and dy.is_cuda:
            dq, dcos, dsin = _rope_backward_triton(dy, q, cos, sin)
            return dq, dcos, dsin
        else:
            assert False, "RoPE1DTritonFn only supports CUDA tensors"
        # # Eager fallback via helper
        # dq, dcos, dsin = rope_backward_eager_fallback(dy, q, cos, sin)
        # return dq, dcos, dsin


def rope_apply_eager_fallback(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Eager fallback for RoPE apply.

    Computes y = q * cos + rotate_half(q) * sin and returns the rotated tensor.
    This function exists to make fallback behavior explicit and testable.
    """
    return (q * cos) + (rotate_half(q) * sin)


def rope_backward_eager_fallback(
    dy: torch.Tensor, q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Eager fallback for RoPE backward.

    Returns gradients (dq, dcos, dsin). Assumes cos/sin are shaped as (b, 1, s, d)
    and broadcast over head dim; reduces grads over that head dim accordingly.

    TODO: If cos/sin shapes differ, extend reduction rules to match broadcasting.
    """
    dq = (dy * cos) - (rotate_half(dy * sin))
    dcos_full = dy * q
    dsin_full = dy * rotate_half(q)

    if dcos_full.dim() == 4 and cos.dim() == 4 and cos.shape[1] == 1:
        dcos = dcos_full.sum(dim=1, keepdim=True)
        dsin = dsin_full.sum(dim=1, keepdim=True)
    else:
        # TODO: adapt reduction rules for other broadcasting schemes
        dcos = dcos_full
        dsin = dsin_full
    return dq, dcos, dsin

class RoPE_M(nn.Module):
    """
    RoPE_M (Rotary Position Embedding) layer for applying rotary embeddings to the input tensor.

    Args:
        dim (int): The dimension of the input tensor.
        base (float): The base for the exponential scaling of the position embeddings.
    """

    def __init__(
        self,
        phys_dim: int = 3,
        d_model: int = 384,
        min_log_freq: float = -5,
        max_log_freq: float = 3,
    ) -> None:
        super().__init__()
        self.m = phys_dim
        self.d_model = d_model
        self.min_log_freq = min_log_freq
        self.max_log_freq = max_log_freq
        assert d_model % (phys_dim * 2) == 0, f"d_model {d_model} must be divisible by 2*phys_dim={2*phys_dim}"
        dim_per_phys = d_model // phys_dim
        # (d_model // (m*2), )
        freqs = sinusoidal_frequencies(dim_per_phys, min_log_freq=min_log_freq, max_log_freq=max_log_freq)
        self.freqs = nn.Parameter(freqs, requires_grad=False)

    def forward(self, fx: torch.Tensor, x: torch.Tensor, return_cos_sin: bool = False):
        """
        Apply the RoPE_M layer to the input tensor.

        Args:
            fx (torch.Tensor): Functional values at the input points, shape (batch_size, heads, seq_len, d_model).
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, m).
            return_cos_sin (bool, optional): Whether to return the cosine and sine tensors. Defaults to False.

        Returns:
            torch.Tensor: Output tensor with rotary embeddings applied.
            If return_cos_sin is True, returns a tuple (output, cos, sin).
        """
        # check input dim
        assert x.shape[-1] == self.m, f"Input tensor x must have last dimension {self.m}, but got {x.shape[-1]}"
        assert fx.shape[-1] == self.d_model, (
            f"Functional tensor fx must have last dimension {self.d_model}, but got {fx.shape[-1]}"
        )
        batch_size, seq_len, phys_dim = x.shape
        assert phys_dim == self.m, f"Input position tensor x has phys_dim={phys_dim}, but expected {self.m}"

        # Ensure sequence length matches between fx and x
        has_head = fx.dim() == 4
        if has_head:
            assert fx.shape[2] == seq_len, f"fx seq_len={fx.shape[2]} mismatches x seq_len={seq_len}"
        else:
            assert fx.shape[1] == seq_len, f"fx seq_len={fx.shape[1]} mismatches x seq_len={seq_len}"

        with torch.autocast(device_type=x.device.type, enabled=False):  # Disable autocast for precision in RoPE_M
            if x.dtype != torch.float32:
                x = x.to(torch.float32)

            if x.ndim == 3: # (bs, seqlen, m) => Assume one head 
                freqs = self.freqs.reshape(1, 1, 1, -1) * x.unsqueeze(-1) # (1,1,1,d_model // (m*2)) * (b, s, m, 1) = (b, s, m, d_model//(m*2))
                freqs = freqs.reshape(batch_size, 1, seq_len, -1) # (b, 1, s, d_model//2)
                freqs = torch.cat([freqs, freqs], dim=-1) # (b, 1, s, d_model)
                cos = torch.cos(freqs)
                sin = torch.sin(freqs)
            else:
                H = fx.shape[1]
                assert x.shape[1] == H and x.ndim == 4 # (bs, head, seqlen, m)
                freqs = self.freqs.reshape(1, 1, 1, 1, -1) * x.unsqueeze(-1) # (1,1,1,1,d_model // (m*2)) * (b, h, s, m, 1) = (b, h, s, m, d_model//(m*2))
                freqs = freqs.reshape(batch_size, H, seq_len, -1) # (b, H, s, d_model//2)
                freqs = torch.cat([freqs, freqs], dim=-1) # (b, H, s, d_model)
                cos = torch.cos(freqs)
                sin = torch.sin(freqs)

        cos, sin = cos.to(fx.dtype), sin.to(fx.dtype) # Safe here.
        has_head = fx.dim() == 4
        if not has_head:
            fx = fx.unsqueeze(1) # (b, 1, s, d_model), h=1
        
        # Note: When x has head dimension (x.ndim == 4), cos and sin will also have head dimension
        # This forces apply_rotary_pos_emb to use the fallback implementation since Triton 
        # doesn't support head-specific cos/sin values
        out = apply_rotary_pos_emb(fx, cos, sin) # (b, h, s, d_model)
        
        if not has_head:
            out = out.squeeze(1) # (b, s, d_model)
        if return_cos_sin:
            return out.contiguous(), cos, sin
        return out.contiguous()


if __name__ == "__main__":
    # Example usage and quick tests
    torch.set_default_device("cuda")
    rope = RoPE_M(phys_dim=3, d_model=384, max_log_freq=6)
    fx = torch.randn(2, 10, 384)  # Batch size 2, sequence length 10, d_model 384
    x = torch.randn(2, 10, 3)  # Batch size 2, sequence length 10, m=3
    output = rope(fx, x)
    print(output.shape)  # Should be (2, 10, 384), ok.

    # Random q, cos, sin for fallback test
    # Shapes: q: (b, h, s, d), cos/sin: (b, 1, s, d)
    b, h, s, d = 2, 3, 5, 8
    q = torch.randn(b, h, s, d)
    freqs = torch.randn(b, 1, s, d)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    # forward fallback
    y_eager = rope_apply_eager_fallback(q, cos, sin)
    print("y_eager shape:", y_eager.shape)

    # backward fallback - check gradients shape only
    dy = torch.randn_like(y_eager)
    dq_eager, dcos_eager, dsin_eager = rope_backward_eager_fallback(dy, q, cos, sin)
    print("dq_eager:", dq_eager.shape, "dcos_eager:", dcos_eager.shape, "dsin_eager:", dsin_eager.shape)

    q = q.requires_grad_()
    cos = cos.requires_grad_()
    sin = sin.requires_grad_()

    # forward
    y = apply_rotary_pos_emb(q, cos, sin)
    print("y shape:", y.shape)

    # backward
    y.backward(dy)
    dq_tr, dcos_tr, dsin_tr = q.grad, cos.grad, sin.grad

    print("q.grad shape:", q.grad.shape)
    print("cos.grad shape:", cos.grad.shape)
    print("sin.grad shape:", sin.grad.shape)

    # diff
    print("dq_eager - q.grad:", torch.allclose(dq_eager, q.grad))
    print("dcos_eager - cos.grad:", torch.allclose(dcos_eager, cos.grad))
    print("dsin_eager - sin.grad:", torch.allclose(dsin_eager, sin.grad))

    y2 = rope_apply_eager_fallback(q, cos, sin)
    print("y2 shape:", y2.shape)
    print("y2 - y:", torch.allclose(y2, y))

    q.grad = None
    cos.grad = None
    sin.grad = None

    y2.backward(dy)
    print("q.grad shape:", q.grad.shape)
    print("cos.grad shape:", cos.grad.shape)
    print("sin.grad shape:", sin.grad.shape)

    print("dq_eager - q.grad:", torch.allclose(dq_eager, q.grad))
    print("|error| = ", torch.norm(dq_eager - q.grad))
    print("dcos_eager - cos.grad:", torch.allclose(dcos_eager, cos.grad))
    print("dsin_eager - sin.grad:", torch.allclose(dsin_eager, sin.grad))

    print("dq_tr - q.grad:", torch.allclose(dq_tr, q.grad))
    print("dcos_tr - cos.grad:", torch.allclose(dcos_tr, cos.grad))
    print("dsin_tr - sin.grad:", torch.allclose(dsin_tr, sin.grad))

    # Performance comparison between Triton and eager fallback
    print("\n=== Performance Comparison ===")
    
    # Use specified dimensions
    b, h, s, d = 64, 8, 256, 48
    print(f"Testing with dimensions: b={b}, h={h}, s={s}, d={d}")
    
    # Create test tensors
    q = torch.randn(b, h, s, d, device="cuda" if torch.cuda.is_available() else "cpu")
    freqs = torch.randn(b, h, s, d, device="cuda" if torch.cuda.is_available() else "cpu")
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    
    # Warmup
    for _ in range(10):
        _ = rope_apply_eager_fallback(q, cos, sin)
        if _TRITON_AVAILABLE and q.is_cuda:
            _ = _rope_apply_triton(q, cos, sin)
    
    # Benchmark eager fallback
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    for _ in range(100):
        _ = rope_apply_eager_fallback(q, cos, sin)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    eager_time = time.time() - start_time
    
    # Benchmark Triton if available
    triton_time = None
    if _TRITON_AVAILABLE and q.is_cuda:
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(100):
            _ = _rope_apply_triton(q, cos, sin)
        torch.cuda.synchronize()
        triton_time = time.time() - start_time
        
        print(f"Eager fallback time: {eager_time:.4f}s")
        print(f"Triton time: {triton_time:.4f}s")
        print(f"Speedup: {eager_time/triton_time:.2f}x")
    else:
        print(f"Eager fallback time: {eager_time:.4f}s")
        print("Triton not available or not on CUDA, skipping comparison")
