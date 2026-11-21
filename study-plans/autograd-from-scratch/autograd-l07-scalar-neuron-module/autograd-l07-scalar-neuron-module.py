import torch

def scalar_neuron_module(
    inputs: torch.Tensor,
    weights: torch.Tensor, 
    bias: torch.Tensor, 
    nonlinear: bool
) -> torch.Tensor:
    """
    Returns a scalar tensor preserving the input dtype and device.
    """
    activation = (inputs * weights).sum() + bias
    output = torch.tanh(activation) if nonlinear else activation
    return output
