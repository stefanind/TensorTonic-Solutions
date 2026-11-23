import torch
import triton
import triton.language as tl


@triton.jit
def flash_attention_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, out_ptr,
    N_Q, N_KV, D, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    pid = tl.program_id(0)

    m_offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offs = tl.arange(0, BLOCK_DMODEL) # captures a whole row
    
    q_offs = m_offs[:, None] * D + d_offs[None, :]
    q_mask = (m_offs[:, None] < N_Q) & (d_offs[None, :] < D)
    q_block = tl.load(Q_ptr + q_offs, mask=q_mask, other=0.0)

    # store intermediate stats 
    m = tl.full((BLOCK_M,), -float('inf'), tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    out = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)

    for n_start in range(tl.cdiv(N_KV, BLOCK_N)):
        n_offs = n_start * BLOCK_N + tl.arange(0, BLOCK_N)

        k_offs = n_offs[:, None] * D + d_offs[None, :]
        k_mask = (n_offs[:, None] < N_KV) & (d_offs[None, :] < D)
        k_tile = tl.load(K_ptr + k_offs, mask=k_mask, other=0.0)

        # M, 1, D * 1, N, D -> M, N, D
        s = tl.sum(q_block[:, None, :] * k_tile[None, :, :], axis=2) * scale
        s = tl.where(n_offs[None, :] < N_KV, s, -float('inf'))

        m_new = tl.maximum(m, tl.max(s, axis=1))
        alpha = tl.exp(m - m_new)

        p = tl.exp(s - m_new[:, None]) 
        l = alpha * l + tl.sum(p, axis=1)

        v_offs = n_offs[:, None] * D + d_offs[None, :]
        v_mask = (n_offs[:, None] < N_KV) & (d_offs[None, :] < D)
        v_tile = tl.load(V_ptr + v_offs, mask=v_mask, other=0.0) # BLOCK_N, BLOCK_D

        # M, N, 1 * 1, N, D
        out = alpha[:, None] * out + tl.sum(p[:, :, None] * v_tile[None, :, :], axis=1)
        
        m = m_new

    result = out / l[:, None]
    
    out_offs = m_offs[:, None] * D + d_offs[None, :]
    out_mask = (m_offs[:, None] < N_Q) & (d_offs[None, :] < D)
    tl.store(out_ptr + out_offs, result, mask=out_mask)


def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, out: torch.Tensor) -> None:
    """Launch the FlashAttention forward kernel: out = softmax(Q K^T / sqrt(D)) V."""
    N_Q, D = Q.shape
    N_KV = K.shape[0]
    scale = 1.0 / (D ** 0.5)
    BLOCK_M = 1
    BLOCK_N = 16
    BLOCK_DMODEL = 1
    while BLOCK_DMODEL < D:
        BLOCK_DMODEL *= 2
    grid = ((N_Q + BLOCK_M - 1) // BLOCK_M,)
    flash_attention_fwd_kernel[grid](
        Q, K, V, out,
        N_Q, N_KV, D, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
    )