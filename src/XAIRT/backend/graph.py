from __future__ import annotations
import torch
import numpy as np
import warnings

# Standardizing internal types for Torch
from XAIRT.backend.types import Tensor, TensorNumpy, Callable, Optimizer

warnings.simplefilter("ignore")

__all__ = ["getLayerIndexByName", "get_gradients", "to_numpy"]

def getLayerIndexByName(model: torch.nn.Module, layername: str) -> int:
    """
    In PyTorch, modules are often nested. This searches the flat 
    named_modules list to find the index.
    """
    for idx, (name, layer) in enumerate(model.named_modules()):
        # named_modules includes the parent 'model' at index 0
        if name == layername:
            return idx
    raise ValueError(f"Layer name: {layername} not found in model.")

def get_gradients(model: torch.nn.Module, 
                  x: torch.Tensor, 
                  desired_labels: torch.Tensor, 
                  compute_loss: Callable) -> torch.Tensor:
    """
    Replaces GradientDescent_useGradientTape.
    Calculates the gradient of the loss with respect to the input x.
    """
    # Ensure x tracks gradients
    if not x.requires_grad:
        x = x.clone().detach().requires_grad_(True)
    
    # Forward pass
    preds = model(x)
    loss = compute_loss(preds, desired_labels)
    
    # Calculate gradients: d(loss)/d(x)
    # create_graph=False unless you need second-order derivatives (Hessians)
    grads = torch.autograd.grad(outputs=loss, inputs=x, 
                                 retain_graph=False, 
                                 create_graph=False)[0]
    
    return grads

def train_step_input(model: torch.nn.Module,
                     x: torch.Tensor,
                     desired_labels: torch.Tensor,
                     compute_loss: Callable,
                     optimizer: Optimizer) -> torch.Tensor:
    """
    PyTorch equivalent of the 'TrainOI' block. 
    Updates the INPUT 'x' using an optimizer.
    """
    # PyTorch optimizers work on lists of tensors
    # We ensure x is the only thing the optimizer sees
    optimizer.zero_grad()
    
    preds = model(x)
    loss = compute_loss(preds, desired_labels)
    loss.backward()
    
    optimizer.step()
    
    return loss.detach()

def to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """
    Converts a Torch Tensor to Numpy, handling CPU/GPU transitions.
    Equivalent to tf_to_numpy.
    """
    if isinstance(x, np.ndarray):
        return x
    
    # .detach() removes from graph, .cpu() moves to RAM, .numpy() converts
    return x.detach().cpu().numpy()
