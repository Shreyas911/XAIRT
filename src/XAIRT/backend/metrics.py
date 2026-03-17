from __future__ import annotations
import torch
import numpy as np
import warnings

warnings.simplefilter("ignore")

__all__ = ["metricF1"]

def metricF1(y_true: torch.Tensor | np.ndarray, y_pred: torch.Tensor | np.ndarray) -> torch.Tensor | float:
    """
    Computes the F1 Score using PyTorch operations.
    Works with both Tensors (on GPU/CPU) and Numpy arrays.
    """
    # 1. Convert to torch tensors if inputs are numpy
    is_numpy = isinstance(y_true, np.ndarray)
    if is_numpy:
        y_true = torch.from_numpy(y_true)
        y_pred = torch.from_numpy(y_pred)

    # Ensure float type for calculations
    y_true = y_true.float()
    y_pred = y_pred.float()
    
    epsilon = 1e-7 # Replaces K.epsilon()

    def recall_m(y_true, y_pred):
        # K.round(K.clip(...)) ensures we are looking at binary classifications
        true_positives = torch.sum(torch.round(torch.clamp(y_true * y_pred, 0, 1)))
        possible_positives = torch.sum(torch.round(torch.clamp(y_true, 0, 1)))
        recall = true_positives / (possible_positives + epsilon)
        return recall

    def precision_m(y_true, y_pred):
        true_positives = torch.sum(torch.round(torch.clamp(y_true * y_pred, 0, 1)))
        predicted_positives = torch.sum(torch.round(torch.clamp(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + epsilon)
        return precision

    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    
    f1_score = 2 * ((precision * recall) / (precision + recall + epsilon))
    
    # Return as the same type as input
    return f1_score.item() if is_numpy else f1_score
