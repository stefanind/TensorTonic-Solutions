import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(tl.cdiv(K, BLOCK_K)):
        tile = k * BLOCK_K + offs_k
        
        a_mask = (offs_m[:, None] < M) & (tile[None, :] < K)
        a_offs = offs_m[:, None] * stride_am + tile[None, :] * stride_ak
        a_tile = tl.load(a_ptr + a_offs, mask=a_mask, other=0.0)
        
        b_mask = (tile[:, None] < K) & (offs_n[None, :] < N)
        b_offs = tile[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_tile = tl.load(b_ptr + b_offs, mask=b_mask, other=0.0)

        acc += tl.dot(a_tile, b_tile, input_precision='ieee')    

    c_offs = offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + c_offs, acc, mask=c_mask)


def solve(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> None:
    """Launch matmul_kernel: out = A @ B."""
    M, K = A.shape
    K2, N = B.shape
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid](
        A, B, out,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )