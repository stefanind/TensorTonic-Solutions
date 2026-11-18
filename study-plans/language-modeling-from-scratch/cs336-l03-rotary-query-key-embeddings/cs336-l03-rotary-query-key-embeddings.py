import torch

def rotary_embed(
    q: torch.Tensor, k: torch.Tensor,
    positions: torch.Tensor, inv_freq: torch.Tensor,
) -> dict:
    """
    Returns q_rotated and k_rotated tensors in a dictionary.
    """
    
    if positions.ndim == 1:
        # positions: (S,)
        theta = positions[:, None] * inv_freq[None, :]
        cos = torch.cos(theta)[None, None, :, :]
        sin = torch.sin(theta)[None, None, :, :]
    
    else:
        # positions: (B, S)
        theta = positions[:, :, None] * inv_freq[None, None, :]
        cos = torch.cos(theta)[:, None, :, :]
        sin = torch.sin(theta)[:, None, :, :]
        
    q_even = q[..., 0::2]
    q_odd = q[..., 1::2]
    k_even = k[..., 0::2]
    k_odd = k[..., 1::2]

    q_even_rot = q_even * cos - q_odd * sin
    k_even_rot = k_even * cos - k_odd * sin
    q_odd_rot = q_even * sin + q_odd * cos
    k_odd_rot = k_even * sin + k_odd * cos

    q_rotated = torch.empty_like(q)
    q_rotated[..., 0::2] = q_even_rot
    q_rotated[..., 1::2] = q_odd_rot

    k_rotated = torch.empty_like(q)
    k_rotated[..., 0::2] = k_even_rot
    k_rotated[..., 1::2] = k_odd_rot

    return {"q_rotated": q_rotated, "k_rotated": k_rotated}

    

    