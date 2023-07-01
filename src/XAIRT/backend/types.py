""" Custom type hinting for XAIRT """

from tensorflow import Tensor
from tensorflow.keras.models import Model
from tensorflow.data import Dataset
from typing import Optional, Sequence, Union, Tuple, TypeAlias, TypeVar, TypedDict
import numpy as np

# DUMMY for NotRequired since it is only available in Python 3.11 and onwards
# from typing import NotRequired # use once you have Python 3.11 env
NotRequired: TypeAlias = Optional # Optional is the closest in meaning to NotRequired for now

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

class LayerDict(TypedDict):
	""" Custom typing hint for layer dict to be given to KerasTrainer and child classes """
	
	size      : int
	activation: Union[str, None]
	use_bias  : NotRequired[Union[bool, None]]

class LossDict(TypedDict):
	""" Custom typing hint for weighted loss functions """

	kind  : list['str']
	weight: list['float']

class AnalysisNormalizeDict:
	""" Custom typing hint for information about normalizing XAI analysis """

	bool_: bool
	kind : NotRequired[str]

class AnalysisStatsDict:
	""" Custom typing hint for statistics of analysis of many samples """ 

	mean: NotRequired[TensorNumpy]
