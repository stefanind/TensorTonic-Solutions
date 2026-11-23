import torch

def column_parallel_mlp(
    x: torch.Tensor,
    weight_shards: list[torch.Tensor], 
    bias_shards: list[torch.Tensor] | None = None
) -> dict:
    """
    Returns a dict: local_activations (tensor list), full_activation (tensor).
    """
    if bias_shards is None:
        bias_shards = [
            torch.zeros(w.shape[1], device=w.device, dtype=w.dtype)
            for w in weight_shards
        ]

    ys = []
    for i, weights in enumerate(weight_shards):
        yr = x @ weights + bias_shards[i]
        ys.append(yr)

    return {
        "local_activations": ys,
        "full_activation": torch.cat(ys, dim=-1)
    }