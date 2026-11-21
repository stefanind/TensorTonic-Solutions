import torch

def neuron_backward(
    inputs: torch.Tensor,
    weights: torch.Tensor, 
    bias: torch.Tensor, 
    upstream_gradient: torch.Tensor
) -> tuple:
    """
    Returns a tuple of tensors: output, input gradients, weight gradients, bias gradient.
    """
    dldy = upstream_gradient
    a = weights @ inputs + bias
    y = torch.tanh(a)

    dyda = dldy * (1 - y**2)
    # dldy dyda dadx AND dadw AND dadb
    dadx = dyda * weights
    dadw = dyda * inputs
    dadb = dyda * 1
        
    return (
        y, dadx, dadw, dadb
    )
