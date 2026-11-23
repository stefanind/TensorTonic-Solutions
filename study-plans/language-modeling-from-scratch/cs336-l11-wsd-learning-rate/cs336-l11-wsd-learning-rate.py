
import torch
import math

def wsd_learning_rate(
    total_steps: int,
    warmup_steps: int,
    stable_steps: int,
    peak_learning_rate: int | float,
    final_ratio: int | float,
) -> torch.Tensor:
    """
    Returns a CPU float64 tensor of shape (total_steps,).
    """
    d = total_steps - warmup_steps - stable_steps

    warmup = (
        peak_learning_rate
        * torch.arange(1, warmup_steps + 1, dtype=torch.float64)
        / warmup_steps
        if warmup_steps > 0
        else torch.empty(0, dtype=torch.float64)
    )

    stable = torch.full(
        (stable_steps,),
        peak_learning_rate,
        dtype=torch.float64,
    )

    if d == 1:
        progress = torch.ones(1, dtype=torch.float64)
    elif d > 1:
        progress = torch.arange(d, dtype=torch.float64) / (d - 1)
    else:
        progress = torch.empty(0, dtype=torch.float64)

    decay = peak_learning_rate * (
        final_ratio
        + (1 - final_ratio)
        * (1 + torch.cos(math.pi * progress)) / 2
    )

    return torch.cat([warmup, stable, decay], dim=0)

            
            