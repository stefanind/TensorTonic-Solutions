import torch

def dense_layer_forward(
    inputs: torch.Tensor, 
    weight_matrix: torch.Tensor, 
    biases: torch.Tensor, 
    nonlinear: bool
) -> torch.Tensor:
    """
    Returns an output vector in neuron order, preserving input dtype and device.
    """
    # inputs is length n
    # W is shape m,n
    # b is length m

    a = weight_matrix @ inputs + biases
    if nonlinear:
        a = torch.tanh(a)

    return a

        