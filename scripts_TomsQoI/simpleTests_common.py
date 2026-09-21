"""
Shared by simpleTests_LRP_keras.py, simpleTests_LRP_torch.py, simpleTests_OI_keras.py and
simpleTests_OI_torch.py, converted from notebooks_TomsQoI/simpleTests.ipynb and
notebooks_TomsQoI/simpleTests_eagerExecution.ipynb

The fake data, the networks, the learning rate schedule and the hand computed LRP that the
notebooks compare the libraries with. Only numpy here, so it is the same for both backends.
"""

import argparse

import numpy as np

LOSSES_CLASSIFICATION = [{'kind': 'categorical_crossentropy', 'weight': 1.0}]
LOSSES_REGRESSION = [{'kind': 'mse', 'weight': 1.0}]
NORMALIZE_SUM = {'bool_': True, 'kind': 'Sum'}

def parse_args(backend, epochs, n_samples):
    p = argparse.ArgumentParser(description=f"simpleTests, {backend} backend.")
    p.add_argument("--out-dir", default=f"simpleTests_output_{backend}", help="the trained models are written here")
    p.add_argument("--n-samples", type=int, default=n_samples)
    p.add_argument("--epochs", type=int, default=epochs, help="use a small number to test the script")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 25
    lrate = initial_lrate * drop**np.floor((1+epoch)/epochs_drop)
    return lrate

# --- Data, the same for both backends. The notebooks used the global numpy random state --------

def one_hot(cls0, cls1):
    y = np.zeros((cls0.shape[0], 2))
    y[:,0] = cls0
    y[:,1] = cls1
    return y

def data_lrp(n_samples, seed):
    """
    x1 + 2 x2 >= 3 is class 0, otherwise class 1. Since x1 + 2 x2 is at most 3 for x in [0,1]^2,
    class 0 is empty and the network always predicts class 1 (pred_class = 1 in the notebook).
    """
    X = np.random.default_rng(seed).random((n_samples, 2))
    y = X @ np.array([1, 2])
    return X, one_hot(y >= 3, y < 3)

def data_regression(n_samples, seed):
    X = np.random.default_rng(seed).random((n_samples, 2))
    return X, X @ np.array([1, 2])

def data_oi(n_samples, seed):
    """Class 0 is x1 + 2 x2 in [1,2], class 1 is the opposite."""
    X = np.random.default_rng(seed).random((n_samples, 2))
    y = X @ np.array([1, 2])
    inside = (y >= 1.) & (y <= 2.)
    return X, one_hot(inside, ~inside)

# --- Networks, bias_constraint is NonNeg() for Keras and 'nonneg' for Torch ----------------------

def layers_lrp(n_features, bias_constraint):
    return [{'size': n_features, 'activation': None     , 'use_bias': None},
            {'size': 2         , 'activation': 'softmax', 'use_bias': True, 'bias_constraint': bias_constraint}]

def layers_regression(n_features):
    return [{'size': n_features, 'activation': None     , 'use_bias': None},
            {'size': 8         , 'activation': 'relu'   , 'use_bias': True, 'l2_w_reg': 10  , 'l2_b_reg': 10},
            {'size': 8         , 'activation': 'relu'   , 'use_bias': True, 'l2_w_reg': 0.01, 'l2_b_reg': 0.01},
            {'size': 1         , 'activation': 'linear' , 'use_bias': True, 'l2_w_reg': 0.01, 'l2_b_reg': 0.01}]

def layers_oi(n_features):
    return [{'size': n_features, 'activation': None     , 'use_bias': None},
            {'size': 8         , 'activation': 'relu'   , 'use_bias': True},
            {'size': 8         , 'activation': 'relu'   , 'use_bias': True},
            {'size': 2         , 'activation': 'softmax', 'use_bias': True}]

# --- LRP by hand, for the network with one Dense layer of 2 inputs (layers_lrp) -------------------

def manual_lrp(x, W, b, pred_class, R_last, bounds = ((-1, 1), (-20, 20), (0.4, 0.6))):
    """
    The relevance of the two inputs, by hand, for the output neuron pred_class.
    x is (1, 2), W (2, n_out), b (n_out,) and R_last the output of the neuron before the softmax.
    LRP-Bounded is done for each of the (low, high) of bounds.

    LRP-A1B0 is done in three ways, see https://github.com/albermax/innvestigate/issues/327: with the bias split between the two inputs (2*bias in the denominator, the
    relevance is conserved, R_last minus the sum is 0), with one bias in the denominator and none in the numerator (the bias keeps
    its share, as it does in innvestigate 2.1.0 lrp.alpha_1_beta_0), and without the bias.
    """

    pos = lambda v: np.maximum(v, 0.0)
    neg = lambda v: np.minimum(v, 0.0)
    w0, w1, bc = W[0,pred_class], W[1,pred_class], b[pred_class]
    x0, x1 = x[0,0], x[0,1]

    def report(label, R_0, R_1):
        print(f"{label}: {R_0}, {R_1}, {R_last-R_0-R_1}")

    # LRP-A1B0, the bias is split between the two inputs
    denominator = x0*pos(w0) + x1*pos(w1) + 2*pos(bc)
    report("Manual LRP-A1B0",
           R_last * (x0*pos(w0) + pos(bc)) / denominator, R_last * (x1*pos(w1) + pos(bc)) / denominator)

    denominator = x0*pos(w0) + x1*pos(w1) + pos(bc)
    report("Manual LRP-A1B0 without bias in numerator and only one bias in denominator",
           R_last * x0*pos(w0) / denominator, R_last * x1*pos(w1) / denominator)

    denominator = x0*pos(w0) + x1*pos(w1)
    report("Manual LRP-A1B0 without bias in both numerator and denominator",
           R_last * x0*pos(w0) / denominator, R_last * x1*pos(w1) / denominator)

    # LRP-WSquare
    denominator = w0**2 + w1**2
    report("Manual LRP-WSquare", R_last * w0**2 / denominator, R_last * w1**2 / denominator)

    # LRP-Bounded, for inputs in [low, high]
    for low, high in bounds:
        denominator = x0*w0 + x1*w1 - low*(pos(w0) + pos(w1)) - high*(neg(w0) + neg(w1))
        report(f"Manual LRP-Bounded with bounds ({low}, {high})",
               R_last * (x0*w0 - low*pos(w0) - high*neg(w0)) / denominator,
               R_last * (x1*w1 - low*pos(w1) - high*neg(w1)) / denominator)

def manual_lrp_z_epsilon(x, W, b, pred_class, R_last, epsilon = 1e-7):
    """Same as manual_lrp for LRP-Z and LRP-Epsilon. The bias is not a part of the numerator."""

    w0, w1, bc = W[0,pred_class], W[1,pred_class], b[pred_class]
    x0, x1 = x[0,0], x[0,1]

    def report(label, denominator):
        R_0, R_1 = R_last * x0*w0 / denominator, R_last * x1*w1 / denominator
        print(f"{label}: {R_0}, {R_1}, {R_last-R_0-R_1}")

    report("Manual LRP-Z without bias in numerator and only one bias in denominator", x0*w0 + x1*w1 + bc)
    report("Manual LRP-Z without bias in both numerator and denominator", x0*w0 + x1*w1)
    report("Manual LRP-Epsilon without bias in numerator and only one bias in denominator", x0*w0 + x1*w1 + bc + epsilon)
    report("Manual LRP-Epsilon without bias in both numerator and denominator", x0*w0 + x1*w1 + epsilon)
