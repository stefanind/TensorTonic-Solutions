import torch
import triton
import triton.language as tl


@triton.jit
def rope_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    N, D,
    BLOCK_SIZE: tl.constexpr,
):
    # x is N, D
    # cos n sin are N, D/2
    # need x split into even and odd
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < D//2

    even_offs = row * D + 2 * cols 
    odd_offs = row * D + 2 * cols + 1
    x_even = tl.load(x_ptr + even_offs, mask=mask, other=0.0)
    x_odd = tl.load(x_ptr + odd_offs, mask=mask, other=0.0)

    
    cs_offset = row * (D//2) + cols
    cos = tl.load(cos_ptr + cs_offset, mask=mask, other=0.0)
    sin = tl.load(sin_ptr + cs_offset, mask=mask, other=0.0)


    res_even = x_even * cos - x_odd * sin 
    res_odd = x_even * sin + x_odd * cos

    tl.store(out_ptr + even_offs, res_even, mask=mask)  
    tl.store(out_ptr + odd_offs, res_odd, mask=mask)
    
    


def solve(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, out: torch.Tensor) -> None:
    """Launch the RoPE kernel: rotate pairs of channels with per-row (cos, sin) tables."""
    N, D = x.shape
    BLOCK_SIZE = 1
    while BLOCK_SIZE < D // 2:
        BLOCK_SIZE *= 2
    if BLOCK_SIZE < 1:
        BLOCK_SIZE = 1
    grid = (N,)
    rope_kernel[grid](x, cos, sin, out, N, D, BLOCK_SIZE=BLOCK_SIZE)