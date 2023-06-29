""" Custom type hinting for XAIRT """

from tensorflow import Tensor
from tensorflow.keras.models import Model
from tensorflow.data import Dataset
from typing import Tuple, TypeVar
import numpy as np

# Alias for type(Keras Model)
kModel: TypeAlias = Model

T = TypeVar("T")
TNumpy = TypeVar("TNumpy", bound=np.generic, covariant=True)

# Alias for numpy arrays, matrices and tensors
VectorNumpy: TypeAlias = np.ndarray[Tuple[int], np.dtype[TNumpy]]
MatrixNumpy: TypeAlias = np.ndarray[Tuple[int, int], np.dtype[TNumpy]]
TensorNumpy: TypeAlias = np.ndarray[Tuple[int, ...], np.dtype[TNumpy]]

# Alias for arguments that can either be a scalar or a list
OptionalList: TypeAlias = Union[T, list[T]]
OptionalSequence: TypeAlias = Union[T, Sequence[T]]
