from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset as TorchDataset
from sklearn.linear_model import LinearRegression
from typing import Optional, Sequence, Union, Tuple, Any
from typing import TypeAlias, TypeVar, TypedDict, Dict, Callable

# Standardizing NotRequired (Standard in Python 3.11, fallback provided)
try:
    from typing import NotRequired
except ImportError:
    NotRequired: TypeAlias = Optional

# Alias for type(Torch Model) - Mapping to kModel to maintain lib compatibility
kModel: TypeAlias = nn.Module

T = TypeVar("T")
TNumpy = TypeVar("TNumpy", bound=np.generic, covariant=True)

# Alias for numpy arrays, matrices and tensors
VectorNumpy: TypeAlias = np.ndarray[Tuple[int], np.dtype[TNumpy]]
MatrixNumpy: TypeAlias = np.ndarray[Tuple[int, int], np.dtype[TNumpy]]
TensorNumpy: TypeAlias = np.ndarray[Tuple[int, ...], np.dtype[TNumpy]]

# PyTorch specific Tensor alias
Tensor: TypeAlias = torch.Tensor

# Alias for arguments that can either be a scalar or a list
OptionalList: TypeAlias = Union[T, list[T]]
OptionalSequence: TypeAlias = Union[T, Sequence[T]]

# Alias for PyTorch Optimizers
Optimizer: TypeAlias = optim.Optimizer
# Dataset can be a Torch Dataset object or a Tuple of Tensors (X, y)
Dataset: TypeAlias = Union[TorchDataset, Tuple[torch.Tensor, torch.Tensor]]

class LayerDict(TypedDict):
    """ Custom typing hint for layer dict to be given to Trainer and child classes """
    size      : int
    activation: Union[str, None]
    use_bias  : NotRequired[bool]
    # Adding PyTorch specific regularization keys seen in previous scripts
    l1_w_reg  : NotRequired[float]
    l2_w_reg  : NotRequired[float]

class LossDict(TypedDict):
    """ Custom typing hint for weighted loss functions """
    kind  : list[str]
    weight: list[float]

class AnalysisNormalizeDict(TypedDict):
    """ Custom typing hint for information about normalizing XAI analysis """
    bool_: bool
    kind : NotRequired[str]

class ModelMetadata(TypedDict):
    """ Custom typing hint for model metadata """
    layers : list[LayerDict]
    losses : OptionalList[LossDict]
    optim  : str
    metrics: list[str]

class TrainMetadata(TypedDict):
    """ Custom typing hint for training metadata """
    batch_size       : int
    epochs           : int
    validation_split : float
    filename         : str
    dirname          : str

class AnalysisStatsDict(TypedDict):
    """ Custom typing hint for statistics of analysis of many samples """ 
    mean: NotRequired[TensorNumpy]

class LetzgusDict(TypedDict):
    """ Custom typing hint for passing Letzgus information around """
    y_ref             : float
    sampleLetzgus     : TensorNumpy
    step_width        : float
    max_it            : int
    method_reg        : str

class XAIMethodsDict(TypedDict):
    """ Custom typing hint for passing XAI methods information around """
    name      : str
    optParams : Dict[str, Any]
    title     : str
