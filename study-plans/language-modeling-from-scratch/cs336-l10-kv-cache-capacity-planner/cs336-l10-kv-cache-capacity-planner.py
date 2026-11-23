def kv_cache_capacity_planner(
    model_parameters: int, 
    bytes_per_parameter: int, 
    num_layers: int,
    sequence_length: int, 
    num_attention_heads: int, 
    num_kv_heads: int,
    head_dim: int, 
    bytes_per_element: int, 
    memory_per_gpu: int,
    reserved_memory: int, 
    memory_bandwidth: int,
) -> dict:
    """
    Returns a dict: model_bytes, kv_bytes_per_request, max_batch_size, seconds_per_token, tokens_per_second.
    """
    model_bytes = bytes_per_parameter * model_parameters
    kv_bytes_per_request = (
        2 * num_layers 
        * sequence_length 
        * num_kv_heads 
        * head_dim 
        * bytes_per_element
    )
    max_batch_size = (
        max(
            0, 
            (memory_per_gpu - reserved_memory - model_bytes) 
            // kv_bytes_per_request
        )
    )

    seconds_per_token = (
        (model_bytes + max_batch_size * kv_bytes_per_request)
        / memory_bandwidth
    ) # total bytes / bytes per second -> seconds 
    # called tokens per second because it is time for a single decode step

    tokens_per_second = (
        max_batch_size / seconds_per_token
        if seconds_per_token > 0 else 0.0 # just in case 
    ) # aggregate throughput for the whole batch

    return {
        "model_bytes": int(model_bytes),
        "kv_bytes_per_request": int(kv_bytes_per_request),
        "max_batch_size": int(max_batch_size),
        "seconds_per_token": float(seconds_per_token),
        "tokens_per_second": float(tokens_per_second),
    }
    