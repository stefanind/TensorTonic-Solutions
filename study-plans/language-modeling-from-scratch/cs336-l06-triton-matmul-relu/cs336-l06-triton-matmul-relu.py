import torch
import triton
import triton.language as tl

@triton.jit
def matmul_relu_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    m_size,
    n_size,
    k_size,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Returns nothing; writes the rectified matrix product through out_ptr.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(tl.cdiv(k_size, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)

        a_offs = offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_mask = (offs_m[:, None] < m_size) & (offs_k[None, :] < k_size)
        a_tile = tl.load(a_ptr + a_offs, mask=a_mask, other=0.0)

        b_offs = offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_mask = (offs_k[:, None] < k_size) & (offs_n[None, :] < n_size)
        b_tile = tl.load(b_ptr + b_offs, mask=b_mask, other=0.0)

        acc += tl.dot(a_tile, b_tile, input_precision="ieee")

    result = tl.maximum(0, acc)
    c_offs = offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    c_mask = (offs_m[:, None] < m_size) & (offs_n[None, :] < n_size)
    tl.store(out_ptr + c_offs, result, mask=c_mask)

        
        
    

def solve(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("matrices must have aligned rank-two shapes")
    if out.shape != (a.shape[0], b.shape[1]):
        raise ValueError("out must have shape (M, N)")
    if not a.is_cuda or not b.is_cuda or not out.is_cuda or a.device != b.device or a.device != out.device:
        raise ValueError("all tensors must be on the same CUDA device")
    if a.dtype != b.dtype or a.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("a and b must have a matching supported dtype")
    if out.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("out has an unsupported dtype")
    m_size, k_size = a.shape
    n_size = b.shape[1]
    if m_size == 0 or n_size == 0:
        return
    if k_size == 0:
        out.zero_()
        return
    kernel_a = a.to(torch.float32) if a.dtype == torch.bfloat16 else a
    kernel_b = b.to(torch.float32) if b.dtype == torch.bfloat16 else b
    kernel_out = torch.empty_like(out, dtype=torch.float32) if out.dtype == torch.bfloat16 else out
    block_m, block_n, block_k = 32, 32, 32
    grid = (triton.cdiv(m_size, block_m), triton.cdiv(n_size, block_n))
    matmul_relu_kernel[grid](
        kernel_a,
        kernel_b,
        kernel_out,
        m_size,
        n_size,
        k_size,
        kernel_a.stride(0),
        kernel_a.stride(1),
        kernel_b.stride(0),
        kernel_b.stride(1),
        kernel_out.stride(0),
        kernel_out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    if kernel_out is not out:
        out.copy_(kernel_out)
