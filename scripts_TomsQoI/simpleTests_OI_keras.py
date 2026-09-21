"""
Optimal input (OI) on fake data with the Keras backend, converted from
notebooks_TomsQoI/simpleTests_eagerExecution.ipynb and the OI part of notebooks_TomsQoI/simpleTests.ipynb

    python scripts_TomsQoI/simpleTests_OI_keras.py --out-dir simpleTests_output_keras
    python scripts_TomsQoI/simpleTests_OI_keras.py --epochs 1 --n-samples 20000    # quick test

A network is trained to say if x1 + 2 x2 is in [1,2] (class 0) or not (class 1). Then the input is changed
by gradient descent, on the loss of the network output against a desired class, until the network
gives that class. Eager execution, which is the default, is needed for the optimizers of Keras.

1. Adam on a norm, the basic example of an optimizer.
2. Gradient descent with Keras_GradientDescent_useGradientTape, from two starting points.
3. The same with a Keras optimizer on a tf.Variable, from two starting points.

The data and the networks are in simpleTests_common.py and are shared with simpleTests_OI_torch.py.
"""

import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

# Append the src directory of the repository to sys.path, then import XAIRT
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from XAIRT import TrainKerasFullyConnectedNN, Keras_GradientDescent_useGradientTape, tf_to_numpy

import simpleTests_common as common

def gradient_descent(model, x0, label, n_iters, lr, print_freq):
    """Plain gradient descent on the input, with the gradient from Keras_GradientDescent_useGradientTape."""

    x = np.array(x0, dtype = np.float32)
    desired_labels = tf.convert_to_tensor(np.array(label, dtype = np.float32))
    bce = keras.losses.BinaryCrossentropy()

    print(f"Desired label : {label}")
    for i in range(n_iters):
        grads = Keras_GradientDescent_useGradientTape(model, tf.convert_to_tensor(x), desired_labels, bce)
        x[0,:] = x[0,:] - lr*np.squeeze(tf_to_numpy(grads))
        if (i+1) % print_freq == 0:
            print(f"Iter {i+1}, Prediction {model.predict(x, verbose = 0)}")

    print(f"Optimal input is : {x}")

def optimizer_descent(model, x0, label, optimizer, n_iters, print_freq):
    """Gradient descent on the input with a Keras optimizer, the input is a tf.Variable."""

    x = tf.Variable(np.array(x0, dtype = np.float32))
    desired_labels = tf.convert_to_tensor(np.array(label, dtype = np.float32))
    bce = keras.losses.BinaryCrossentropy()

    print(f"Desired label : {label}")
    for i in range(n_iters):
        with tf.GradientTape() as g:
            g.watch(x)
            loss = bce(desired_labels, model(x))
        # This has to be outside the with statement for efficiency, unless you want higher order derivatives.
        grads = g.gradient(loss, x)
        optimizer.apply_gradients(zip([grads], [x]))
        if (i+1) % print_freq == 0:
            print(f"Iter {i+1}, OI {x.numpy()}")

    print(f"Optimal input is : {x.numpy()}")

if __name__ == "__main__":

    args = common.parse_args("keras", epochs = 5, n_samples = 1000000)

    keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()

    print("Running on GPU" if tf.config.list_physical_devices('GPU') else "Running on CPU")
    os.makedirs(args.out_dir, exist_ok = True)

    ### Basic example of optimization with Adam ###

    x = tf.Variable(np.array([1., -1.], dtype = np.float32))
    adam = keras.optimizers.Adam(learning_rate = 0.1)
    for i in range(10):
        with tf.GradientTape() as g:
            g.watch(x)
            loss = tf.norm(x)
        adam.apply_gradients(zip([g.gradient(loss, x)], [x]))
        print(x.numpy())

    ### Optimal input for a classification problem ###

    X, oneHot = common.data_oi(args.n_samples, args.seed)
    print(f"Samples per class: {oneHot.sum(axis = 0)}")

    keras.backend.clear_session()
    sgd = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    K = TrainKerasFullyConnectedNN(X, oneHot, layers = common.layers_oi(X.shape[1]), losses = common.LOSSES_CLASSIFICATION,
                                   optimizer = sgd, metrics = ['mae'], batch_size = 2048, epochs = args.epochs,
                                   validation_split = 0.2, filename = 'model_simpleTests_OI', dirname = args.out_dir,
                                   random_nn_seed = args.seed, verbose = 2)
    best_model = K.quickTrain(common.step_decay)

    # The corner (0, 0) to class 0, then the optimal input of class 0 (0.3, 0.6) to class 1
    print("Gradient descent with Keras_GradientDescent_useGradientTape")
    gradient_descent(best_model, [[0., 0.]], [[1., 0.]], 100, 0.01, 10)
    gradient_descent(best_model, [[0.3, 0.6]], [[0., 1.]], 100, 0.01, 10)

    print("Gradient descent with a Keras optimizer on a tf.Variable")
    optimizer_descent(best_model, [[0.0, 0.1]], [[1., 0.]], keras.optimizers.SGD(learning_rate = 0.01), 2000, 100)
    optimizer_descent(best_model, [[5.0, 10.0]], [[1., 0.]], keras.optimizers.SGD(learning_rate = 0.01), 2000, 100)
