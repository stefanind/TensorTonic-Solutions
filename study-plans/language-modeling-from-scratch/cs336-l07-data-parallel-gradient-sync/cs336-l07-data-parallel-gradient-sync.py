import torch

def data_parallel_gradient_sync(
    parameter_replicas: list, 
    gradient_replicas: list, 
    learning_rate: float
) -> dict:
    """
    Returns a dict: mean_gradients (tensor list), updated_replicas (rank-ordered tensor lists).
    """
    world_size = len(parameter_replicas)
    parameter_count = len(parameter_replicas[0])
   
    mean_grads = []
    for i, param in enumerate(parameter_replicas[0]):
        grads = [rank[i] if rank[i] is not None else torch.zeros_like(param)
                 for rank in gradient_replicas
                ]
        mean_grad = torch.stack(grads).float().mean(dim=0)
        mean_grads.append(mean_grad.to(param.dtype)) # list of tensors

    updated_replicas = []
    for _ in range(world_size):
        # params is a list of parameter tensors
        updated_params = []
        for i, param in enumerate(parameter_replicas[0]):
            updated = (
                param.float() - learning_rate * mean_grads[i]  
            ).to(param.dtype)

            updated_params.append(updated.clone()) # clone because each param has its own independent storage

        updated_replicas.append(updated_params)
   
    
    return {
        "mean_gradients": mean_grads,
        "updated_replicas": updated_replicas
    }
