"""
LRP on fake data with the Keras backend (TensorFlow, innvestigate), converted from the LRP and the
linear regression parts of notebooks_TomsQoI/simpleTests.ipynb

    python scripts_TomsQoI/simpleTests_LRP_keras.py --out-dir simpleTests_output_keras
    python scripts_TomsQoI/simpleTests_LRP_keras.py --epochs 2 --n-samples 500    # quick test

1. A network with a single Dense layer of 2 inputs and 2 softmax outputs, explained with LRP-A1B0 and with
   the input layer rules of innvestigate, next to the same relevance computed by hand.
2. A network for a linear regression, explained with LRP-Z and LRP-A1B0, next to the explanation of a
   linear regression (XLR).

The data, the networks and the hand computed LRP are in simpleTests_common.py and are shared with
simpleTests_LRP_torch.py. The OI part of the notebook is in simpleTests_OI_keras.py. In the first part R_last - sum(a) of a line is
not 0 for lrp.alpha_1_beta_0, since innvestigate gives the bias a share of the relevance, see LRP_manual_MWE_keras.py and
https://github.com/albermax/innvestigate/issues/327. init(), setup() and
print_analyzers() are also used by LRP_manual_MWE_keras.py and understandingLRP_keras.py.
"""

import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.constraints import NonNeg
import innvestigate

# Append the src directory of the repository to sys.path, then import XAIRT
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from XAIRT import TrainKerasFullyConnectedNN, TrainLR, XAIKeras, XLR

import simpleTests_common as common

def a1b0_analyzers(bounds = ()):
    """
    (label, innvestigate name, kwargs) of LRP-A1B0, and of A1B0_IB that ignores the bias, with the default input
    layer rule, WSquare, Bounded and Bounded with each of the (low, high) of bounds. The input layer rule replaces
    the rule of the first layer.
    """
    analyzers = []
    rules = [("default", None), ("WSquare", "WSquare"), ("Bounded", "Bounded")] + [(f"bounds {b}", b) for b in bounds]
    for label, rule in rules:
        for name in ('lrp.alpha_1_beta_0', 'lrp.alpha_1_beta_0_IB'):
            analyzers.append((f"{name}, {label}", name, {} if rule is None else {'input_layer_rule': rule}))
    return analyzers

Z_EPSILON_ANALYZERS = [("lrp.z"                    , 'lrp.z'      , {}),
                       ("lrp.epsilon"              , 'lrp.epsilon', {}),
                       ("lrp.epsilon, without bias", 'lrp.epsilon', {'bias': False})]

def init(args):
    """Seeds, and eager execution off, which innvestigate needs."""

    ### https://stackoverflow.com/questions/36288235/how-to-get-stable-results-with-tensorflow-setting-random-seed ###
    ### https://keras.io/examples/keras_recipes/reproducibility_recipes/ ###
    keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()

    # Required by innvestigate
    tf.compat.v1.disable_eager_execution()

    print("Running on GPU" if tf.config.list_physical_devices('GPU') else "Running on CPU")
    os.makedirs(args.out_dir, exist_ok = True)

def train(x, y, layers, losses, args, name, decay_func = None):

    keras.backend.clear_session()
    sgd = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

    K = TrainKerasFullyConnectedNN(x, y, layers = layers, losses = losses, optimizer = sgd,
                                   metrics = ['mae'], batch_size = 128, epochs = args.epochs,
                                   validation_split = 0.2, filename = name, dirname = args.out_dir,
                                   random_nn_seed = args.seed, verbose = 2)
    return K.quickTrain(decay_func)

def setup(args):
    """
    Trains the network with a single Dense layer and returns it without its softmax, one sample x (1, 2), the weights
    W (2, 2) and the bias b (2,) of the layer, the predicted class and its output before the softmax.
    """

    X, oneHot = common.data_lrp(args.n_samples, args.seed)
    best_model = train(X, oneHot, common.layers_lrp(X.shape[1], NonNeg()), common.LOSSES_CLASSIFICATION,
                       args, 'model_simpleTests_LRP', common.step_decay)
    model_wo_softmax = innvestigate.model_wo_softmax(best_model)

    idx = 2
    x = X[np.newaxis, idx]
    y_true = model_wo_softmax.predict(x)
    pred_class = int(np.argmax(y_true[0]))
    W, b = model_wo_softmax.layers[1].get_weights()
    R_last = float(y_true[0,pred_class])

    print(f"x @ W + b - output of the network: {x @ W + b - y_true}")
    print(f"Output {y_true}, W {W}, b {b}, x {x}")

    return model_wo_softmax, x, W, b, pred_class, R_last

def print_analyzers(model_wo_softmax, x, R_last, analyzers):
    """The relevance of x of each analyzer, and how far the sum of it is from R_last."""
    for label, name, kwargs in analyzers:
        a = innvestigate.create_analyzer(name, model_wo_softmax, **kwargs).analyze(x)
        print(f"{label}: {a}, {R_last - np.sum(a)}")

if __name__ == "__main__":

    args = common.parse_args("keras", epochs = 100, n_samples = 10000)
    init(args)

    ### Understanding LRP ###

    model_wo_softmax, x, W, b, pred_class, R_last = setup(args)
    print_analyzers(model_wo_softmax, x, R_last, a1b0_analyzers(bounds = [(13, 9)]))
    common.manual_lrp(x, W, b, pred_class, R_last)

    ### Linear regression using an ANN ###

    X, y = common.data_regression(args.n_samples, args.seed)
    best_model = train(X, y[:,np.newaxis], common.layers_regression(X.shape[1]), common.LOSSES_REGRESSION,
                       args, 'model_simpleTests_regr')
    regr = TrainLR(X, y, y_ref = 0.0, fit_intercept = False).quickTrain()

    lrp_methods = [dict(name='lrp.z'             , optParams = {}, title = 'LRP-Z'),
                   dict(name='lrp.alpha_1_beta_0', optParams = {}, title = 'LRP-A1B0')]

    for method in lrp_methods:
        a, _ = XAIKeras(best_model, method, 'classic', X, common.NORMALIZE_SUM).quick_analyze()
        print(f"{method['name']}: {np.mean(a, axis = 0)}")

    a, _ = XLR(regr, X).quick_analyze()
    print(f"lrp.LR: {np.mean(a, axis = 0)}")
