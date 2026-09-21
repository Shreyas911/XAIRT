"""
LRP on fake data with the PyTorch backend (Captum), the counterpart of simpleTests_LRP_keras.py, which is
converted from the LRP and the linear regression parts of notebooks_TomsQoI/simpleTests.ipynb

    python scripts_TomsQoI/simpleTests_LRP_torch.py --out-dir simpleTests_output_torch
    python scripts_TomsQoI/simpleTests_LRP_torch.py --epochs 2 --n-samples 500    # quick test

The data, the networks and the hand computed LRP are in simpleTests_common.py and are shared with the
Keras script, so the two backends run the identical experiment. init(), setup() and print_analyzers()
are also used by LRP_manual_MWE_torch.py and understandingLRP_torch.py. In the first part R_last - sum(a) shows how Captum treats
the bias of LRP-A1B0, which innvestigate gives a share of the relevance, see LRP_manual_MWE_keras.py and
https://github.com/albermax/innvestigate/issues/327.

Captum only has the plain alpha=1, beta=0 rule and the epsilon rule. The 'IB' variants and the input layer rules
of innvestigate (WSquare, Bounded), and LRP-Epsilon without the bias, do not exist there, so of those only the
relevance computed by hand is printed.
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
from XAIRT import TrainTorchFullyConnectedNN, TrainLR, XAITorch, XLR, model_wo_softmax_torch

import simpleTests_common as common

# The same names as the Keras script, the ones that Captum has
A1B0_ANALYZERS = [dict(name='lrp.alpha_1_beta_0', optParams = {}, title = 'LRP-A1B0')]
Z_EPSILON_ANALYZERS = [dict(name='lrp.z'      , optParams = {}, title = 'LRP-Z'),
                       dict(name='lrp.epsilon', optParams = {}, title = 'LRP-Epsilon')]

def init(args):

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Running on GPU" if torch.cuda.is_available() else "Running on CPU")
    os.makedirs(args.out_dir, exist_ok = True)

def train(x, y, layers, losses, args, name, decay_func = None):

    T = TrainTorchFullyConnectedNN(x, y, layers = layers, losses = losses,
                                   optimizer = 'sgd',    # nesterov, momentum 0.9
                                   learning_rate = 0.01,
                                   metrics = [],         # ignored
                                   batch_size = 128, epochs = args.epochs,
                                   validation_split = 0.2, filename = name, dirname = args.out_dir,
                                   random_nn_seed = args.seed, verbose = 1)
    return T.quickTrain(decay_func)

def setup(args):
    """
    Trains the network with a single Linear layer and returns it without its softmax, one sample x (1, 2), the weights
    W (2, 2) and the bias b (2,) of the layer, the predicted class and its output before the softmax.
    """

    X, oneHot = common.data_lrp(args.n_samples, args.seed)
    best_model = train(X, oneHot, common.layers_lrp(X.shape[1], 'nonneg'), common.LOSSES_CLASSIFICATION,
                       args, 'model_simpleTests_LRP', common.step_decay)
    # Captum LRP does not support the final Softmax, the layers are shared with best_model
    model_wo_softmax = model_wo_softmax_torch(best_model)

    idx = 2
    x = X[np.newaxis, idx]
    with torch.no_grad():
        y_true = model_wo_softmax(torch.from_numpy(x).float()).numpy()
    pred_class = int(np.argmax(y_true[0]))
    linear = model_wo_softmax[0]
    # A Torch weight is (out, in), a Keras kernel (in, out)
    W = linear.weight.detach().numpy().T
    b = linear.bias.detach().numpy()
    R_last = float(y_true[0,pred_class])

    print(f"x @ W + b - output of the network: {x @ W + b - y_true}")
    print(f"Output {y_true}, W {W}, b {b}, x {x}")

    return model_wo_softmax, x, W, b, pred_class, R_last

def print_analyzers(model_wo_softmax, x, R_last, methods):
    """The relevance of x of each method, and how far the sum of it is from R_last."""
    for method in methods:
        a, _ = XAITorch(model_wo_softmax, method, 'classic', x, {'bool_': False}).quick_analyze()
        print(f"{method['name']}: {a}, {R_last - np.sum(a)}")

if __name__ == "__main__":

    args = common.parse_args("torch", epochs = 100, n_samples = 10000)
    init(args)

    ### Understanding LRP ###

    model_wo_softmax, x, W, b, pred_class, R_last = setup(args)
    print_analyzers(model_wo_softmax, x, R_last, A1B0_ANALYZERS)
    common.manual_lrp(x, W, b, pred_class, R_last)

    ### Linear regression using an ANN ###

    X, y = common.data_regression(args.n_samples, args.seed)
    best_model = train(X, y[:,np.newaxis], common.layers_regression(X.shape[1]), common.LOSSES_REGRESSION,
                       args, 'model_simpleTests_regr')
    regr = TrainLR(X, y, y_ref = 0.0, fit_intercept = False).quickTrain()

    lrp_methods = [dict(name='lrp.z'             , optParams = {}, title = 'LRP-Z'),
                   dict(name='lrp.alpha_1_beta_0', optParams = {}, title = 'LRP-A1B0')]

    for method in lrp_methods:
        a, _ = XAITorch(best_model, method, 'classic', X, common.NORMALIZE_SUM).quick_analyze()
        print(f"{method['name']}: {np.mean(a, axis = 0)}")

    a, _ = XLR(regr, X).quick_analyze()
    print(f"lrp.LR: {np.mean(a, axis = 0)}")
