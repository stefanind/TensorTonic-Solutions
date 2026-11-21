import torch

def linear_attention_duality(
    q: torch.Tensor, 
    k: torch.Tensor,
    v: torch.Tensor
) -> dict:
    """
    Returns a dict of tensors: 
    parallel_output, 
    recurrent_output, 
    final_state.
    """
    # parallel
    parallel_scores = q @ k.transpose(-1, -2)
    t = q.shape[1]
    mask = torch.triu(
        torch.ones((t, t), device=q.device, dtype=torch.bool), 
        diagonal=1
    )
    parallel_scores.masked_fill_(mask, 0.0) # 0 because no softmax

    B, T, Dk = q.shape
    Dv = v.shape[-1]

    state = torch.zeros(
        (B, Dk, Dv), device=q.device, dtype=q.dtype
    )
    outputs = []
    for t in range(T):
        Q = q[:, t, :].float()  # [B, Dk]
        K = k[:, t, :].float()  # [B, Dk]
        V = v[:, t, :].float()  # [B, Dv]
        update = K[:, :, None] * V[:, None, :]
        state = state + update # B, Dk, Dv
        # B, 1, Dk @ B, Dk, Dv
        y = Q.unsqueeze(1) @ state # B, 1, Dv
        outputs.append(y.squeeze(1))

    out = torch.stack(outputs, dim=1)
        

    return {
        "parallel_output": (parallel_scores @ v).to(q.dtype),
        "recurrent_output": out.to(q.dtype),
        "final_state": state.to(q.dtype)
    }
