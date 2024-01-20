from XAIRT.backend.types import TensorNumpy
import numpy as np
from scipy.stats import pearsonr

__all__ = ["correlation"]

def correlation(X: TensorNumpy, 
                y: TensorNumpy) -> TensorNumpy:

    correlations = np.zeros((X.shape[1],), dtype = float)
    
    for i in range(X.shape[1]):
        correlations[i], _ = pearsonr(X[:,i], y)

    return correlations