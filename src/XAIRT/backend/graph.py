""" Low-level operations with Keras """
from __future__ import annotations

import warnings
import numpy as np
warnings.simplefilter("ignore")

from XAIRT.backend.types import Tensor, TensorNumpy
from XAIRT.backend.types import Callable, Optimizer, Variable

import tensorflow.keras as keras
import tensorflow as tf

# Get useful types from types.py
from XAIRT.backend.types import kModel

__all__ =["getLayerIndexByName", "GradientDescent_useGradientTape", 
          "TrainOI_useGradientTape", "tf_to_numpy"]

def getLayerIndexByName(model: kModel, layername: str) -> int:
	for idx, layer in enumerate(model.layers):
		if layer.name == layername: 
			return idx
	raise ValueError(f"layername: {layername} not found.")

# Vanilla gradient descent
@tf.function # Crucial function decorator for speedup
def GradientDescent_useGradientTape(model: kModel, 
					x: Tensor,
					desired_labels: Tensor,
					compute_loss: Callable) -> Tensor:

    with tf.GradientTape() as g:
        g.watch(x)
        preds = model(x)
        loss = compute_loss(desired_labels, preds)
    
    # This has to be outside the with statement for efficiency, unless you want higher order derivatives.
    grads = g.gradient(loss, x)

    return grads

### NOTE: Requires eager execution to be ENABLED
@tf.function # Crucial function decorator for speedup
def TrainOI_useGradientTape(model: kModel,
					  x: Variable,
					  desired_labels: Variable,
					  compute_loss: Callable,
                      optimizer: Optimizer) -> None:

    # Because apply_gradient takes list of (gradient, variable) pairs, 
    # input must be a instance of tf.Variable rather than tf.Tensor.
    # https://github.com/tensorflow/tensorflow/issues/31273
    # https://colab.research.google.com/github/tensorflow/docs/blob/snapshot-keras/site/en/guide/keras/writing_a_training_loop_from_scratch.ipynb
    with tf.GradientTape() as g:
        g.watch(x)
        preds = model(x)
        loss = compute_loss(desired_labels, preds)

    # This has to be outside the with statement for efficiency, unless you want higher order derivatives.
    grads = g.gradient(loss, x)
    # Apply gradient using in-built tf optimizer
    optimizer.apply_gradients(zip([grads], [x]))

    return loss

# Tf to Numpy when eager execution is disabled, which is the case for LRP.
def tf_to_numpy(x: Tensor) -> TensorNumpy:
    return np.array(tf.keras.backend.get_value(x))