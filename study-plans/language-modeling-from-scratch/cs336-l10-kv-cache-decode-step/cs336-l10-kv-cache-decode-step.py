import torch

def kv_cache_decode_step(
    query: torch.Tensor, 
    new_key: torch.Tensor, 
    new_value: torch.Tensor,
    key_cache: torch.Tensor, 
    value_cache: torch.Tensor,
    num_query_heads: int, 
    num_kv_heads: int,
) -> dict:
    """
    Returns a dict of tensors: output, new_key_cache, new_value_cache.
    """
    device = query.device
    B, Hq, D = query.shape
    B, Hkv, S, D = key_cache.shape
    group_size = Hq // Hkv
    keys = torch.cat(
        [key_cache, new_key.unsqueeze(2)],
        dim=2
    )  # [B, Hkv, S+1, D]
    
    values = torch.cat(
        [value_cache, new_value.unsqueeze(2)],
        dim=2
    )
    
    # Create a separate broadcasted view
    k = keys.unsqueeze(2)  # [B, Hkv, 1, S+1, D]
    
    q = query.reshape(B, Hkv, group_size, 1, D)
    
    scores = (q @ k.transpose(-1, -2)) * (D ** -0.5)
    att = torch.softmax(scores, dim=-1)
    
    out = att @ values.unsqueeze(2)
    out = out.squeeze(-2).reshape(B, Hq, D)
    
    return {
        "output": out,
        "new_key_cache": keys,
        "new_value_cache": values,
    }   
    
    
    
    
