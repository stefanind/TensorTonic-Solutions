import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, out_ptr, x_row_stride, out_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):

    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    x_offs = row * x_row_stride + cols
    mask = cols < n_cols

    x = tl.load(x_ptr + x_offs, mask=mask, other=-float('inf'))

    maximum = tl.max(x, axis=0)
    numerator = tl.exp(x - maximum)
    denominator = 1.0 / tl.sum(numerator, axis=0)
    softmax = numerator * denominator

    o_offs = row * out_row_stride + cols
    tl.store(out_ptr + o_offs, softmax, mask=mask)


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch softmax_kernel with one program per row."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    softmax_kernel[grid](
        x, out, x.stride(0), out.stride(0), N, BLOCK_SIZE=BLOCK_SIZE,
    )