import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_bwd_kernel(
    x_ptr, gamma_ptr, dy_ptr,
    dx_ptr, dgamma_ptr, dbeta_ptr,
    stride_x_row, stride_dy_row, stride_dx_row,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
): 
    # dx.shape = x.shape ; 
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # load each vector
    # get x
    x_offs = row * stride_x_row + cols # if x.stride == dx.stride then can use same
    x = tl.load(x_ptr + x_offs, mask=mask, other=0.0)
    # get dy
    dy_offs = row * stride_dy_row + cols
    dy = tl.load(dy_ptr + dy_offs, mask=mask, other=0.0)
    # get gamma
    gamma = tl.load(gamma_ptr + cols, mask=mask, other=0.0)

    # calculate xhat
    mu = tl.sum(x, axis=0) / N # axis 0 because a program only has one row ; so mu is a scalar
    diff = tl.where(mask, x - mu, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = tl.rsqrt(var + eps)
    xhat = diff * rstd

    # calculate dydxhat
    dydxhat = dy * gamma # broadcasts properly
    # calculate dydgamma
    dydgamma = dy * xhat # one row so no reduction yet 
    # calculate dydbeta
    dydbeta = dy # atomic add to sum across i
    # calculate dx, which is dydxhat * dxhatdx
    dx = rstd * (
        dydxhat
        - tl.sum(dydxhat, axis=0) / N
        - xhat 
        * tl.sum(dydxhat * xhat, axis=0) / N
    )
    
    dx_offs = row * stride_dx_row + cols
    tl.store(dx_ptr + dx_offs, dx, mask=mask)
    tl.atomic_add(dbeta_ptr + cols, dy, mask=mask)
    tl.atomic_add(dgamma_ptr + cols, dydgamma, mask=mask)

    
    
    
    
    
    

    
    
    
    
    


def solve(
    x: torch.Tensor, gamma: torch.Tensor, dy: torch.Tensor,
    dx_out: torch.Tensor, dgamma_out: torch.Tensor, dbeta_out: torch.Tensor,
    eps: float,
) -> None:
    """Launch the LayerNorm backward kernel: one program per row, atomic reductions for dgamma and dbeta."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    dgamma_out.zero_()
    dbeta_out.zero_()
    grid = (M,)
    layernorm_bwd_kernel[grid](
        x, gamma, dy,
        dx_out, dgamma_out, dbeta_out,
        x.stride(0), dy.stride(0), dx_out.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )