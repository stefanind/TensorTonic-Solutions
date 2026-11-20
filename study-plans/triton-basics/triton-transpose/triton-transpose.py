import torch
import triton
import triton.language as tl


@triton.jit
def transpose_kernel(
    a_ptr, out_ptr,
    M, N,
    stride_am, stride_an,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # load
    a_offset = rows[:, None] * stride_am + cols[None, :] * stride_an
    a_mask = (rows[:, None] < M) & (cols[None, :] < N)
    a = tl.load(a_ptr + a_offset, mask=a_mask, other=0.0)

    # store ; stride is switched because transpose
    o_offset = rows[:, None] * stride_on + cols[None, :] * stride_om
    tl.store(out_ptr + o_offset, a, mask=a_mask)
    
    


def solve(A: torch.Tensor, out: torch.Tensor) -> None:
    """Launch transpose_kernel: out[j, i] = A[i, j]."""
    M, N = A.shape
    BLOCK_M = 32
    BLOCK_N = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    transpose_kernel[grid](
        A, out,
        M, N,
        A.stride(0), A.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )