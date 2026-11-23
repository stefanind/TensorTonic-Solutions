import torch

def squared_error_loss_gradients(
    predictions: torch.Tensor, 
    targets: torch.Tensor
) -> tuple:
    """
    Returns a scalar loss tensor and a prediction-shaped gradient tensor.
    """
    gradients = torch.zeros_like(predictions)

    diff = predictions - targets
    
    loss = torch.sum(diff * diff)

    gradients = 2 * diff

    return loss, gradients

    
