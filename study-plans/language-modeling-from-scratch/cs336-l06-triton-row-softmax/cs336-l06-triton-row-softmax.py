import torch
import triton
import triton.language as tl

@triton.jit
def row_softmax_kernel(x_ptr, out_ptr, row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    """
    Returns nothing; writes normalized rows through out_ptr.
    """
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    offset = pid * row_stride + cols
    mask = cols < n_cols
    x = tl.load(x_ptr + offset, mask=mask, other=-float('inf')).to(tl.float32)
    maximum = tl.max(x, axis=0)
    numerator = tl.exp(x - maximum)
    denominator = 1.0 / tl.sum(tl.exp(x - maximum), axis=0)
    result = numerator * denominator

    out_offset = pid * n_cols + cols
    tl.store(out_ptr + out_offset, result, mask=mask)
    
    
    
    

def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError("x must be a nonempty-width matrix")
    if x.shape != out.shape or x.device != out.device or x.dtype != out.dtype:
        raise ValueError("x and out must have matching shape, device, and dtype")
    if not x.is_cuda or x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("x must be a supported CUDA tensor")
    if x.stride(1) != 1 or not out.is_contiguous():
        raise ValueError("input columns and output must be contiguous")
    if x.shape[1] > 65536:
        raise ValueError("row width exceeds the supported limit")
    if x.shape[0] == 0:
        return
    kernel_x = x
    kernel_out = out
    if x.dtype == torch.bfloat16:
        kernel_x = x.to(torch.float32)
        kernel_out = torch.empty_like(x, dtype=torch.float32)
    block_size = triton.next_power_of_2(x.shape[1])
    grid = (x.shape[0],)
    row_softmax_kernel[grid](
        kernel_x,
        kernel_out,
        kernel_x.stride(0),
        x.shape[1],
        BLOCK_SIZE=block_size,
    )
    if kernel_out is not out:
        out.copy_(kernel_out)
