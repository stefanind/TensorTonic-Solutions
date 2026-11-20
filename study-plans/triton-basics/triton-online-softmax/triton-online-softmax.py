import torch
import triton
import triton.language as tl


@triton.jit
def online_softmax_kernel(
    x_ptr, out_ptr,
    M, N,
    BLOCK_SIZE: tl.constexpr,
):

    row = tl.program_id(0) * N
    cols = tl.arange(0, BLOCK_SIZE)

    m = -float('inf')
    l = 0.0

    for tile in range(tl.cdiv(N, BLOCK_SIZE)):
        x_offs = tile * BLOCK_SIZE + cols 
        x_mask = x_offs < N
        x = tl.load(x_ptr + row + x_offs, mask=x_mask, other=-float('inf'))
        m_new = tl.maximum(m, tl.max(x, axis=0))
        l = l * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new

    for tile in range(tl.cdiv(N, BLOCK_SIZE)):
        offs = tile * BLOCK_SIZE + cols 
        mask = offs < N
        x = tl.load(x_ptr + row + offs, mask=mask, other=-float('inf'))
        numerator = tl.exp(x - m)
        denominator = 1.0 / l
        softmax = numerator * denominator
        tl.store(out_ptr + row + offs, softmax, mask=mask)
        
    
    

    
    


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch the online softmax kernel row-wise."""
    M, N = x.shape
    BLOCK_SIZE = 1024
    grid = (M,)
    online_softmax_kernel[grid](x, out, M, N, BLOCK_SIZE=BLOCK_SIZE)