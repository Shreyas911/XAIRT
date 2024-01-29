""" Low-level operations with Keras """
from __future__ import annotations

import warnings
import numpy as np
warnings.simplefilter("ignore")

from XAIRT.backend.types import Tensor, Callable, TensorNumpy

import tensorflow.keras as keras
import tensorflow as tf

# Get useful types from types.py
from XAIRT.backend.types import kModel

__all__ =["getLayerIndexByName", "useGradientTape", "tf_to_numpy"]

def getLayerIndexByName(model: kModel, layername: str) -> int:
	for idx, layer in enumerate(model.layers):
		if layer.name == layername: 
			return idx
	raise ValueError(f"layername: {layername} not found.")

# Useful function for Optimal Input calculations
@tf.function # Crucial function decorator for speedup
def useGradientTape(model: kModel, 
					x: Tensor,
					desired_labels: Tensor,
					compute_loss: Callable) -> Tensor:

    with tf.GradientTape(persistent=True) as g:
        g.watch(x)
        preds = model(x)
        loss = compute_loss(desired_labels, preds)
        # Expand dimensions of scalar loss to (1,1)
        loss = tf.expand_dims(tf.expand_dims(loss, axis=(0,)), axis=(0,))
    
    # This has to be outside the with statement for efficiency, unless you want higher order derivatives.
    grads = g.batch_jacobian(loss, x)

    return grads

# Tf to Numpy when eager execution is disabled, which is the case for LRP.
def tf_to_numpy(x: Tensor) -> TensorNumpy:
    return np.array(tf.keras.backend.get_value(x))