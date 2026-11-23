import torch
import triton
import triton.language as tl


@triton.jit
def grouped_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    programs_per_group = num_pid_n * GROUP_SIZE_M # pid 1
    group_id = pid // programs_per_group # 1 // 8 -> 0
    local_id = pid % programs_per_group # 1 % 8 -> 1
    first_pid_m = group_id * GROUP_SIZE_M # 0 * 2 -> 0
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M) # 8 - 0, 2 -> 2
    pid_n = local_id // group_size_m # 1 // 2 -> 0
    pid_m = first_pid_m + local_id % group_size_m # 0 + 1 % 2 -> 1

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(tl.cdiv(K, BLOCK_K)):
        k_tile = k * BLOCK_K + offs_k
        
        a_offs = offs_m[:, None] * stride_am + k_tile[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (k_tile[None, :] < K)        
        a_tile = tl.load(a_ptr + a_offs, mask=a_mask, other=0.0)
        
        b_offs = k_tile[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_mask = (k_tile[:, None] < K) & (offs_n[None, :] < N)
        b_tile = tl.load(b_ptr + b_offs, mask=b_mask, other=0.0)

        acc += tl.dot(a_tile, b_tile, input_precision='ieee')

    c_offs = offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + c_offs, acc, mask=c_mask)    
        


def solve(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> None:
    """Launch grouped_matmul_kernel with a 1D grid and grouped pid remap."""
    M, K = A.shape
    K2, N = B.shape
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    GROUP_SIZE_M = 8
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    grouped_matmul_kernel[grid](
        A, B, out,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )