"""
Optimal input (OI) on fake data with the PyTorch backend, the counterpart of simpleTests_OI_keras.py, which is
converted from notebooks_TomsQoI/simpleTests_eagerExecution.ipynb and the OI part of
notebooks_TomsQoI/simpleTests.ipynb

    python scripts_TomsQoI/simpleTests_OI_torch.py --out-dir simpleTests_output_torch
    python scripts_TomsQoI/simpleTests_OI_torch.py --epochs 1 --n-samples 20000    # quick test

A network is trained to say if x1 + 2 x2 is in [1,2] (class 0) or not (class 1). Then the input is changed
by gradient descent, on the loss of the network output against a desired class, until the network
gives that class.

1. Adam on a norm, the basic example of an optimizer.
2. Gradient descent with Torch_GradientDescent_useAutograd, from two starting points.
3. The same with a torch.optim optimizer on the input tensor, from two starting points.

The data and the networks are in simpleTests_common.py and are shared with the Keras script.
"""

import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
# XAIRT imports TensorFlow too. Keep it off the GPU, which is for PyTorch.
tf.config.set_visible_devices([], 'GPU')
import torch

# Append the src directory of the repository to sys.path, then import XAIRT
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from XAIRT import TrainTorchFullyConnectedNN, Torch_GradientDescent_useAutograd, torch_to_numpy

import simpleTests_common as common

def gradient_descent(model, x0, label, n_iters, lr, print_freq):
    """Plain gradient descent on the input, with the gradient from Torch_GradientDescent_useAutograd."""

    x = torch.tensor(x0, dtype = torch.float32)
    desired_labels = torch.tensor(label, dtype = torch.float32)
    bce = torch.nn.BCELoss()

    print(f"Desired label : {label}")
    for i in range(n_iters):
        grads = Torch_GradientDescent_useAutograd(model, x, desired_labels, bce)
        x = x - lr*grads
        if (i+1) % print_freq == 0:
            with torch.no_grad():
                print(f"Iter {i+1}, Prediction {torch_to_numpy(model(x))}")

    print(f"Optimal input is : {torch_to_numpy(x)}")

def optimizer_descent(model, x0, label, optimizer_class, lr, n_iters, print_freq):
    """Gradient descent on the input with a torch.optim optimizer."""

    x = torch.tensor(x0, dtype = torch.float32, requires_grad = True)
    desired_labels = torch.tensor(label, dtype = torch.float32)
    optimizer = optimizer_class([x], lr = lr)
    bce = torch.nn.BCELoss()

    print(f"Desired label : {label}")
    for i in range(n_iters):
        optimizer.zero_grad()
        loss = bce(model(x), desired_labels)
        loss.backward()
        optimizer.step()
        if (i+1) % print_freq == 0:
            print(f"Iter {i+1}, OI {torch_to_numpy(x)}")

    print(f"Optimal input is : {torch_to_numpy(x)}")

if __name__ == "__main__":

    args = common.parse_args("torch", epochs = 5, n_samples = 1000000)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Running on GPU" if torch.cuda.is_available() else "Running on CPU")
    os.makedirs(args.out_dir, exist_ok = True)

    ### Basic example of optimization with Adam ###

    x = torch.tensor([1., -1.], requires_grad = True)
    adam = torch.optim.Adam([x], lr = 0.1)
    for i in range(10):
        adam.zero_grad()
        torch.norm(x).backward()
        adam.step()
        print(torch_to_numpy(x))

    ### Optimal input for a classification problem ###

    X, oneHot = common.data_oi(args.n_samples, args.seed)
    print(f"Samples per class: {oneHot.sum(axis = 0)}")

    T = TrainTorchFullyConnectedNN(X, oneHot, layers = common.layers_oi(X.shape[1]), losses = common.LOSSES_CLASSIFICATION,
                                   optimizer = 'sgd',    # nesterov, momentum 0.9
                                   learning_rate = 0.01, metrics = [], batch_size = 2048, epochs = args.epochs,
                                   validation_split = 0.2, filename = 'model_simpleTests_OI', dirname = args.out_dir,
                                   random_nn_seed = args.seed, verbose = 1)
    best_model = T.quickTrain(common.step_decay)

    # The corner (0, 0) to class 0, then the optimal input of class 0 (0.3, 0.6) to class 1
    print("Gradient descent with Torch_GradientDescent_useAutograd")
    gradient_descent(best_model, [[0., 0.]], [[1., 0.]], 100, 0.01, 10)
    gradient_descent(best_model, [[0.3, 0.6]], [[0., 1.]], 100, 0.01, 10)

    print("Gradient descent with torch.optim.SGD on the input")
    optimizer_descent(best_model, [[0.0, 0.1]], [[1., 0.]], torch.optim.SGD, 0.01, 2000, 100)
    optimizer_descent(best_model, [[5.0, 10.0]], [[1., 0.]], torch.optim.SGD, 0.01, 2000, 100)
