from XAIRT.backend.types import TensorNumpy
import numpy as np
from scipy.stats import pearsonr
import torch

__all__ = ["correlation", "correlation_torch"]

def correlation(X: TensorNumpy, y: TensorNumpy) -> TensorNumpy:
    """
    Standard Numpy/Scipy implementation. 
    Maintains compatibility with your existing XAI analysis loops.
    """
    # Ensure inputs are numpy arrays in case Torch tensors were passed
    if hasattr(X, 'detach'): X = X.detach().cpu().numpy()
    if hasattr(y, 'detach'): y = y.detach().cpu().numpy()

    correlations = np.zeros((X.shape[1],), dtype=float)
    
    for i in range(X.shape[1]):
        # pearsonr returns (correlation, p-value), we take the first element
        correlations[i], _ = pearsonr(X[:, i], y)

    return correlations

def correlation_torch(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    A PyTorch-native version that can run on GPU for faster 
    computation on very large feature sets.
    """
    # Standardize shapes
    X = X.float()
    y = y.float().view(-1, 1)
    
    # Center the data
    X_mean = torch.mean(X, dim=0)
    y_mean = torch.mean(y)
    
    X_centered = X - X_mean
    y_centered = y - y_mean
    
    # Compute correlation: (cov(X,y)) / (std(X) * std(y))
    numerator = torch.mm(X_centered.t(), y_centered).squeeze()
    denominator = torch.sqrt(torch.sum(X_centered**2, dim=0) * torch.sum(y_centered**2))
    
    return numerator / (denominator + 1e-8)
