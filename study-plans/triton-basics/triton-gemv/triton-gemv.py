import torch
import triton
import triton.language as tl


@triton.jit
def gemv_kernel(
    a_ptr, x_ptr, out_ptr,
    M, N,
    stride_am, stride_an,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for start in range(tl.cdiv(N, BLOCK_N)):
        tile = start * BLOCK_N + cols
        mask = (rows[:, None] < M) & (tile[None, :] < N)
        a_offs = rows[:, None] * stride_am + tile[None, :] * stride_an
        a_tile = tl.load(a_ptr + a_offs, mask=mask, other=0.0)
        x_tile = tl.load(x_ptr + tile, mask=tile < N, other=0.0)
        acc += tl.sum(a_tile * x_tile, axis=1)
        
    tl.store(out_ptr + rows, acc, mask=rows < M)
        
        


def solve(A: torch.Tensor, x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch gemv_kernel: out = A @ x."""
    M, N = A.shape
    BLOCK_M = 32
    BLOCK_N = 64
    grid = (triton.cdiv(M, BLOCK_M),)
    gemv_kernel[grid](
        A, x, out,
        M, N,
        A.stride(0), A.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )