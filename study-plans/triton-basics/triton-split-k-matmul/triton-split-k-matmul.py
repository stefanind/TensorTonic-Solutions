import torch
import triton
import triton.language as tl


@triton.jit
def split_k_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = pid % SPLIT_K
    pid_n = (pid // SPLIT_K) % tl.cdiv(N, BLOCK_N)
    pid_m = pid // (SPLIT_K * tl.cdiv(N, BLOCK_N))

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # how many k are there in each program?
    # if we have to split the K by SPLIT_K 
    # then how many k are there in each program
    k_per_split = tl.cdiv(K, SPLIT_K)
    # where is the program starting? 
    # we skip over the amount of k_per_split for each program
    k_start = pid_k * k_per_split
    # where is the program ending?
    # a program either ends at the k_per_split (plus the starting point)
    # or at the maximum allowed amount K
    k_end = tl.minimum(k_start + k_per_split, K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_base in range(k_start, k_end, BLOCK_K):
        k_indices = k_base + offs_k

        a_offs = offs_m[:, None] * stride_am + k_indices[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (k_indices[None, :] < k_end)
        a_tile = tl.load(a_ptr + a_offs, mask=a_mask, other=0.0)

        b_offs = k_indices[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_mask = (k_indices[:, None] < k_end) & (offs_n[None, :] < N)
        b_tile = tl.load(b_ptr + b_offs, mask=b_mask, other=0.0)

        acc += tl.dot(a_tile, b_tile, input_precision="ieee")

    c_offs = offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.atomic_add(c_ptr + c_offs, acc, mask=c_mask)

    


def solve(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> None:
    """Launch split_k_matmul_kernel with SPLIT_K K-partitions and atomic_add."""
    M, K = A.shape
    K2, N = B.shape
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    SPLIT_K = 4
    out.zero_()
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N) * SPLIT_K,)
    split_k_matmul_kernel[grid](
        A, B, out,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        SPLIT_K=SPLIT_K,
    )