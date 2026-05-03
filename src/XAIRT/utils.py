from __future__ import annotations

import warnings
import numpy as np
from scipy.stats import pearsonr
warnings.simplefilter("ignore")

import tensorflow.keras as keras
import tensorflow as tf
from keras import backend as K

from tf import Tensor
from keras import Model

from beartype import beartype
from jaxtyping import Float
from collections.abc import Callable

__all__ = ["getLayerIndexByName",
           "Keras_GradientDescent_useGradientTape",
           "tf_to_numpy", "metricF1", "correlation"]

@beartype
def getLayerIndexByName(model: Model, layername: str) -> int:
    for idx, layer in enumerate(model.layers):
        if layer.name == layername: 
            return idx
    raise ValueError(f"layername: {layername} not found.")

# Crucial function decorator for speedup
@tf.function
@beartype
def Keras_GradientDescent_useGradientTape(model: Model,
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

# tf to numpy when eager execution is disabled, which is the case for LRP.
@beartype
def tf_to_numpy(x: Tensor) -> Float[np.ndarray, "..."]:
    return np.array(tf.keras.backend.get_value(x))

# https://datascience.stackexchange.com/questions/105101/which-keras-metric-for-multiclass-classification
@beartype
def metricF1(y_true: Float[np.ndarray, "dimy dimx"],
             y_pred: Float[np.ndarray, "dimy dimx"]) -> float:

    def recall_m(y_true: Float[np.ndarray, "dimy dimx"],
                 y_pred: Float[np.ndarray, "dimy dimx"]) -> float:
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = TP / (Positives+K.epsilon())    
        return recall 

    def precision_m(y_true: Float[np.ndarray, "dimy dimx"],
                    y_pred: Float[np.ndarray, "dimy dimx"]) -> float:
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = TP / (Pred_Positives+K.epsilon())
        return precision 

    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))

@beartype
def correlation(X: Float[np.ndarray, "dimy dimx"], 
                y: Float[np.ndarray, "dimy"]) -> Float[np.ndarray, "dimx"]:

    correlations = np.zeros((X.shape[1],), dtype = float)
    
    for i in range(X.shape[1]):
        correlations[i], _ = pearsonr(X[:,i], y)

    return correlations

