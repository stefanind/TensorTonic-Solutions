import torch
import triton
import triton.language as tl


@triton.jit
def cross_entropy_kernel(
    logits_ptr, target_ptr, loss_out_ptr,
    stride_logits_row,
    B, C,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    row = pid * stride_logits_row + cols
    mask = cols < C
    # no tiles because block_size will always be greater than the col amouunt

    logits = tl.load(logits_ptr + row, mask=mask, other=-float('inf'))
    maximum = tl.max(logits, axis=0)
    exp = tl.exp(logits - maximum)
    lse = tl.log(tl.sum(exp, axis=0)) + maximum # AXIS IS 0 BECAUSE A PROGRAM IS ONLY ONE ROW

    target_id = tl.load(target_ptr + pid)
    target_logit = tl.load(logits_ptr + pid * stride_logits_row + target_id)
    
    row_loss = lse - target_logit

    tl.atomic_add(loss_out_ptr, row_loss)


def solve(logits: torch.Tensor, target: torch.Tensor, loss_out: torch.Tensor) -> None:
    """Launch the cross-entropy kernel: one program per row, atomic accumulate, then divide by B."""
    B, C = logits.shape
    BLOCK_SIZE = triton.next_power_of_2(C)
    loss_out.zero_()
    grid = (B,)
    cross_entropy_kernel[grid](
        logits, target, loss_out,
        logits.stride(0),
        B, C,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    loss_out.div_(B)