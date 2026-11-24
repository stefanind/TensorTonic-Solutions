import torch

def speculative_sampling(
    draft_tokens: torch.Tensor,
    draft_probabilities: torch.Tensor,
    target_probabilities: torch.Tensor, 
    acceptance_random_values: torch.Tensor,
    residual_random_value: int | float,
) -> dict:
    """
    Returns a dict: tokens (int64 tensor), accepted_count (int), rejected_at (int or None).
    """

    positions = torch.arange(draft_tokens.numel(), device=draft_tokens.device)
    q = draft_probabilities[positions, draft_tokens]
    p = target_probabilities[positions, draft_tokens]

    acceptance = torch.clamp(p / q, max=1.0)
    accepted = acceptance_random_values < acceptance

    rejected_indices = positions[~accepted]
    if rejected_indices.numel() > 0:
        rejected_at = rejected_indices[0].item()
        accepted_count = rejected_at
    else:
        rejected_at = None
        accepted_count = draft_tokens.numel()
        
    tokens = draft_tokens[:accepted_count]
    
    if rejected_at is not None:
        q_dist = draft_probabilities[rejected_at]
        p_dist = target_probabilities[rejected_at]

        residual = torch.clamp(p_dist - q_dist, min=0.0)
        residual = residual / residual.sum()

        cdf = residual.cumsum(dim=0)

        replacement = torch.searchsorted(
            cdf, residual_random_value, right=True 
        )

        tokens = torch.cat([tokens, replacement.reshape(1)], dim=0)
    
    return {
        "tokens": tokens,
        "accepted_count": accepted_count,
        "rejected_at": rejected_at
    }
        
    
