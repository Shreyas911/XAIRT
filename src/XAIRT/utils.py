from __future__ import annotations

import warnings
import numpy as np
from scipy.stats import pearsonr
warnings.simplefilter("ignore")

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import torch

from beartype import beartype
from jaxtyping import Float
from collections.abc import Callable

__all__ = ["getLayerIndexByName", "getLayerIndexByName_torch",
           "Keras_GradientDescent_useGradientTape", "Torch_GradientDescent_useAutograd",
           "tf_to_numpy", "torch_to_numpy",
           "metricF1", "metricF1_torch",
           "model_wo_softmax_torch", "keras_to_torch",
           "correlation", "correlation_torch"]

@beartype
def getLayerIndexByName(model: Model, layername: str) -> int:
    for idx, layer in enumerate(model.layers):
        if layer.name == layername: 
            return idx
    raise ValueError(f"layername: {layername} not found.")

# Index among the direct children, like model.layers in Keras (no root offset).
# Note: a Keras Dense layer includes its activation, but in a Torch nn.Sequential
# the Linear and its activation are separate children, so indices do not line up
# across backends. Sequential children are named "0", "1", ...
@beartype
def getLayerIndexByName_torch(model: torch.nn.Module, layername: str) -> int:
    for idx, (name, _) in enumerate(model.named_children()):
        if name == layername:
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

# Gradient of the loss w.r.t. the input x. Follows the Torch loss convention,
# compute_loss(preds, desired_labels), whereas the Keras one uses (y_true, y_pred).
@beartype
def Torch_GradientDescent_useAutograd(model: torch.nn.Module,
                                      x: torch.Tensor,
                                      desired_labels: torch.Tensor,
                                      compute_loss: Callable) -> torch.Tensor:
    x = x.detach().clone().requires_grad_(True)

    preds = model(x)
    loss = compute_loss(preds, desired_labels)

    # No graph is kept, unless you want higher order derivatives.
    grads, = torch.autograd.grad(loss, x)

    return grads

# tf to numpy when eager execution is disabled, which is the case for LRP.
@beartype
def tf_to_numpy(x: Tensor) -> Float[np.ndarray, "..."]:
    return np.array(tf.keras.backend.get_value(x))

# torch to numpy, also for tensors that require grad or live on a GPU.
@beartype
def torch_to_numpy(x: torch.Tensor) -> Float[np.ndarray, "..."]:
    return x.detach().cpu().numpy()

# https://datascience.stackexchange.com/questions/105101/which-keras-metric-for-multiclass-classification
# Used as a Keras metric (metrics=[metricF1], custom_objects={'metricF1': metricF1}),
# so it is called with symbolic tensors and must NOT be type-checked with beartype
# as numpy arrays or a float.
def metricF1(y_true, y_pred):

    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = TP / (Positives+K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = TP / (Pred_Positives+K.epsilon())
        return precision 

    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# Same computation as metricF1 (K.epsilon() = 1e-7). Not a training metric, since the
# Torch trainer ignores metrics, so unlike metricF1 it can be type-checked and return a float.
@beartype
def metricF1_torch(y_true: Float[torch.Tensor, "dimy dimx"],
                   y_pred: Float[torch.Tensor, "dimy dimx"]) -> float:

    epsilon = 1e-7

    TP = torch.sum(torch.round(torch.clamp(y_true * y_pred, 0, 1)))
    Positives = torch.sum(torch.round(torch.clamp(y_true, 0, 1)))
    Pred_Positives = torch.sum(torch.round(torch.clamp(y_pred, 0, 1)))

    recall = TP / (Positives + epsilon)
    precision = TP / (Pred_Positives + epsilon)

    return (2*((precision*recall)/(precision+recall+epsilon))).item()

# Torch counterpart of innvestigate.model_wo_softmax, needed for Captum LRP, which
# supports neither Softmax nor Sigmoid. The layers (and weights) are shared, not copied.
# Unlike innvestigate's, this also drops a final Sigmoid (a single-output binary model).
@beartype
def model_wo_softmax_torch(model: torch.nn.Sequential) -> torch.nn.Sequential:

    layers = list(model.children())
    if layers and isinstance(layers[-1], (torch.nn.Softmax, torch.nn.Sigmoid)):
        layers = layers[:-1]

    return torch.nn.Sequential(*layers)

# Copies the weights of a fully connected Keras model into a Torch nn.Sequential, so that both XAI
# libraries can explain exactly the same network. A Keras Dense kernel is (in, out), the weight of a
# Torch Linear is (out, in), hence the transpose.
@beartype
def keras_to_torch(model: Model) -> torch.nn.Sequential:

    activations = {"relu"   : torch.nn.ReLU,
                   "sigmoid": torch.nn.Sigmoid,
                   "tanh"   : torch.nn.Tanh,
                   "softmax": lambda: torch.nn.Softmax(dim=1),
                   "linear" : None}

    modules = []
    for layer in model.layers:

        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        if not isinstance(layer, tf.keras.layers.Dense):
            raise NotImplementedError(f"Layer {layer.name} is a {type(layer).__name__}, only Dense is supported.")

        name = layer.activation.__name__
        if name not in activations:
            raise NotImplementedError(f"Activation '{name}' is not supported.")

        weights = layer.get_weights()
        linear = torch.nn.Linear(weights[0].shape[0], weights[0].shape[1], bias = layer.use_bias)
        with torch.no_grad():
            linear.weight.copy_(torch.from_numpy(weights[0].T.copy()))
            if layer.use_bias:
                linear.bias.copy_(torch.from_numpy(weights[1].copy()))
        modules.append(linear)

        if activations[name] is not None:
            modules.append(activations[name]())

    return torch.nn.Sequential(*modules).eval()

@beartype
def correlation(X: Float[np.ndarray, "dimy dimx"],
                y: Float[np.ndarray, "dimy"]) -> Float[np.ndarray, "dimx"]:

    correlations = np.zeros((X.shape[1],), dtype = float)

    for i in range(X.shape[1]):
        correlations[i], _ = pearsonr(X[:,i], y)

    return correlations

@beartype
def correlation_torch(X: Float[torch.Tensor, "dimy dimx"],
                      y: Float[torch.Tensor, "dimy"]) -> Float[torch.Tensor, "dimx"]:

    # Standardize shapes
    X = X.float()
    y = y.float().reshape(-1, 1)

    # Center the data
    X_mean = torch.mean(X, dim=0)
    y_mean = torch.mean(y)

    X_centered = X - X_mean
    y_centered = y - y_mean

    # Compute correlation: (cov(X,y)) / (std(X) * std(y))
    # squeeze(1), not squeeze(): the latter returns a 0-d tensor when dimx == 1.
    numerator = torch.mm(X_centered.t(), y_centered).squeeze(1)
    denominator = torch.sqrt(torch.sum(X_centered**2, dim=0) * torch.sum(y_centered**2))

    # Exact Pearson, as in scipy's pearsonr: no epsilon, so a constant column
    # gives nan in both backends instead of 0 here.
    return numerator / denominator

