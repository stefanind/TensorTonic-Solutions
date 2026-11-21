import torch
import torch.nn.functional as F

def parameter_matched_swiglu(
    x: torch.Tensor,
    w_g: torch.Tensor, 
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    base_params: int,
) -> dict:
    """
    Returns output, hidden_width, and parameter_count in a dictionary.
    """
    D = x.shape[-1]
    H = w_g.shape[-1]
    h = min(
        H, max(
            1, math.floor(base_params / (3*D) + 0.5)
        )
    )

    w_g = w_g[:, :h]
    w_v = w_v[:, :h]
    w_o = w_o[:h, :]

    y = (F.silu(x @ w_g) * (x @ w_v)) @ w_o

    return {
        "output": y,
        "hidden_width": h,
        "parameter_count": 3 * D * h
    } 

    
