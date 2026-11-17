import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_fwd_kernel(
    x_ptr, gamma_ptr, beta_ptr, out_ptr,
    stride_x_row, stride_out_row,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    # for each input row of matrix M,N compute mean and variance
    # across FEATURE dimension dim=0, i.e., one value for each row
    # one program per row, therefore loop over tile for each row

    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)

    total = 0.0

    for tile in range(tl.cdiv(N, BLOCK_SIZE)):
        cols_indices = tile * BLOCK_SIZE + cols
        mask = cols_indices < N
        offset = pid * stride_x_row + cols_indices
        x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        total += tl.sum(x, axis=0)

    mu = total / N

    variance_total = 0.0

    for tile in range(tl.cdiv(N, BLOCK_SIZE)):
        cols_indices = tile * BLOCK_SIZE + cols
        mask = cols_indices < N
        offset = pid * stride_x_row + cols_indices
        x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        squared_diff = tl.where(mask, (x - mu) * (x - mu), 0.0)
        variance_total += tl.sum(squared_diff, axis=0)
        
    sigma_squared = variance_total / N

    inv_std = tl.rsqrt(sigma_squared + eps)

    for tile in range(tl.cdiv(N, BLOCK_SIZE)):
        cols_indices = tile * BLOCK_SIZE + cols
        mask = cols_indices < N
        offset = pid * stride_x_row + cols_indices
        x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        norm = (x - mu) * inv_std

        # gamma and beta size j
        gamma = tl.load(gamma_ptr + cols_indices, mask=mask, other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + cols_indices, mask=mask, other=0.0).to(tl.float32)
        result = norm * gamma + beta

        out_offset = pid * stride_out_row + cols_indices
        
        tl.store(out_ptr + out_offset, result, mask=mask)        


def solve(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, out: torch.Tensor, eps: float) -> None:
    """Launch the LayerNorm forward kernel: one program per row."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    layernorm_fwd_kernel[grid](
        x, gamma, beta, out,
        x.stride(0), out.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )