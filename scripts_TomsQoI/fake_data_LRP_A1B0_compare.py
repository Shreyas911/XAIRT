"""
The quick version of eccov4r5_LRP_A1B0_compare.py: no ECCO data, a fake classification problem with a known
answer, so it runs in a minute and can be used to check that the libraries agree.

    python scripts_TomsQoI/fake_data_LRP_A1B0_compare.py --out-dir LRP_output_fake
    python scripts_TomsQoI/fake_data_LRP_A1B0_compare.py --positive-inputs        # every layer has inputs >= 0
    python scripts_TomsQoI/fake_data_LRP_A1B0_compare.py --epochs 5 --n-explain 20    # quick test

LRP-A1B0 (alpha=1, beta=0 in every layer) and LRP-A1B0-IB (the same, ignoring the bias), the raw relevance of the same
samples, in these cases:

  keras             : network trained with Keras, explained with innvestigate
  keras_torch       : the network of 'keras', its weights copied into PyTorch (keras_to_torch), explained with Captum
  keras_torch_split : the same with XAITorch's optParams {'sign_split': True}, innvestigate's rule as a Captum rule
  torch             : network trained with PyTorch, explained with Captum. A different network, not expected to agree as well
  numpy_split       : the network of 'keras', by hand with the positive contributions (x_i w_ij)+
  numpy_wplus       : the same with x_i (w_ij)+

The two libraries differ in one thing only, the positive contributions: (x_i w_ij)+ in innvestigate and x_i (w_ij)+ in
Captum. Both put the raw bias in the denominator and pass the relevance through the ReLUs as it is. The two are the same for
inputs >= 0, the hidden layers after a ReLU, but not for the first layer with the standard normal X. Found by reading
innvestigate 2.1.2 and Captum 0.9.0, and confirmed by this script (job 3544480, see HANDOFF.md): keras = keras_torch_split =
numpy_split and keras_torch = numpy_wplus to float precision for any input, keras = keras_torch only with --positive-inputs.

Also LRP-A1B0-B, the method of the ECCO scripts (Bounded rule in the first layer, A1B0 ignoring the bias in the others), with
innvestigate's default bounds (-1, 1) and with the training range: innvestigate, XAITorch's BoundedRule on the same network and
the formula by hand, which are expected to agree to float precision.

The data: X is standard normal and the class is the sign of a linear combination of only the first --n-informative features
plus noise, so the relevance should sit on those features. 'informative_share' is the share of the relevance that they get,
to be compared with n_informative/n_features for a network that ignores them. With --positive-inputs X is uniform in [0,1],
and the score is centered.

Written to --out-dir: agreement_fake.json and the models.
"""

import argparse
import json
import os
import sys
from os.path import join
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import innvestigate
import torch

# Append the src directory of the repository to sys.path, then import XAIRT
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from XAIRT import (TrainKerasFullyConnectedNN, TrainTorchFullyConnectedNN,
                   XAIKeras, XAITorch, keras_to_torch, model_wo_softmax_torch)

METHOD_NAMES = {'a1b0': 'lrp.alpha_1_beta_0', 'a1b0_ib': 'lrp.alpha_1_beta_0_IB'}
RAW = {'bool_': False}
LOSSES = [{'kind': 'categorical_crossentropy', 'weight': 1.0}]
BATCH_SIZE = 64
TEST_FRAC = 0.2
VAL_FRAC = 0.2

# (case, case): what agreement or disagreement of the pair says
PAIRS = [('keras'            , 'numpy_split'),          # innvestigate does what the reading says
         ('keras_torch'      , 'numpy_wplus'),          # Captum does what the reading says
         ('keras_torch_split', 'numpy_split'),          # the sign_split rule is right
         ('keras'            , 'keras_torch_split'),    # same network, same rule, two libraries
         ('keras'            , 'keras_torch'),          # same network, each library with its own rule
         ('keras'            , 'torch')]                # two networks

def parse_args():
    p = argparse.ArgumentParser(description="LRP-A1B0 with innvestigate, Captum and by hand, fake data.")
    p.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "results" / Path(sys.argv[0]).stem),
                   help="models and results are written here, by default scripts_TomsQoI/results/<script name>")
    p.add_argument("--n-samples", type=int, default=3000)
    p.add_argument("--n-features", type=int, default=40)
    p.add_argument("--n-informative", type=int, default=4, help="the first features, the only ones that matter")
    p.add_argument("--n-explain", type=int, default=200, help="number of test samples that are explained")
    p.add_argument("--epochs", type=int, default=100, help="use a small number to test the script")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--positive-inputs", action="store_true", help="X in [0,1], so that every layer has inputs >= 0")
    return p.parse_args()

def make_data(n_samples, n_features, n_informative, seed, positive_inputs = False):
    """X (samples, features) and the one-hot class (samples, 2), the same layout as the ECCO data."""

    rng = np.random.default_rng(seed)
    if positive_inputs:
        X = rng.random((n_samples, n_features)).astype(np.float32)
    else:
        X = rng.standard_normal((n_samples, n_features)).astype(np.float32)

    w = np.zeros(n_features)
    w[:n_informative] = rng.uniform(1.0, 2.0, n_informative) * rng.choice([-1.0, 1.0], n_informative)
    score = X @ w + 0.5*rng.standard_normal(n_samples)
    if positive_inputs:
        score -= np.median(score)    # the classes would not be balanced otherwise

    y = np.zeros((n_samples, 2), dtype = np.float32)
    y[:,0] = score >= 0.0
    y[:,1] = score <  0.0

    return X, y

def make_layers(n_features):
    return [{'size': n_features, 'activation': None     , 'use_bias': None},
            {'size': 16        , 'activation': 'relu'   , 'use_bias': True},
            {'size': 8         , 'activation': 'relu'   , 'use_bias': True},
            {'size': 2         , 'activation': 'softmax', 'use_bias': True}]

def train_keras(x_t, y_t, x_v, y_v, layers, args, models_dir):

    keras.backend.clear_session()
    sgd = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

    K = TrainKerasFullyConnectedNN(x_t, y_t, layers = layers, losses = LOSSES, optimizer = sgd,
                                   metrics = ['accuracy'], batch_size = BATCH_SIZE, epochs = args.epochs,
                                   filename = 'model_fake', dirname = models_dir,
                                   validation_data = (x_v, y_v), random_nn_seed = args.seed, verbose = 0)
    return K.quickTrain()

def train_torch(x_t, y_t, x_v, y_v, layers, args, models_dir):

    T = TrainTorchFullyConnectedNN(x_t, y_t, layers = layers, losses = LOSSES, optimizer = 'sgd',
                                   learning_rate = 0.01, metrics = [], batch_size = BATCH_SIZE, epochs = args.epochs,
                                   filename = 'model_fake', dirname = models_dir,
                                   validation_data = (x_v, y_v), random_nn_seed = args.seed, verbose = 0)
    return T.quickTrain()

def predict_torch(model, X):
    with torch.no_grad():
        return model(torch.from_numpy(X).float()).numpy()

def dense_layers(keras_model_wo_softmax):
    """(W (in, out), b (out,) or None, activation name) of each Dense layer, in float64."""
    layers = []
    for layer in keras_model_wo_softmax.layers:
        if isinstance(layer, keras.layers.Dense):
            weights = layer.get_weights()
            b = weights[1].astype(np.float64) if layer.use_bias else None
            layers.append((weights[0].astype(np.float64), b, layer.activation.__name__))
    return layers

def numpy_a1b0(layers, x, sign_split, ignore_bias, bounds = None):
    """
    LRP-A1B0 of one sample x (features,) for the output neuron with the largest value, by hand:
        R_i = sum_j c_ij / (sum_i' c_i'j + b_j) R_j,   c_ij = (x_i w_ij)+ if sign_split, else x_i (w_ij)+
    b_j is the raw bias (0 if ignore_bias); a zero denominator gives zero relevance. With bounds (low, high) the first
    layer has the Bounded rule instead, without the bias: c_ij = x_i w_ij - low w_ij+ - high w_ij-.
    """

    # Forward pass, keeping the input of each layer
    inputs, a = [], x.astype(np.float64)
    for W, b, act in layers:
        inputs.append(a)
        a = a @ W + (0.0 if b is None else b)
        if act == 'relu':
            a = np.maximum(a, 0.0)
        elif act != 'linear':
            raise NotImplementedError(act)

    R = np.zeros_like(a)
    R[np.argmax(a)] = a[np.argmax(a)]

    for k, (W, b, _), a_in in zip(reversed(range(len(layers))), reversed(layers), reversed(inputs)):
        if k == 0 and bounds is not None:
            low, high = bounds
            c = a_in[:,None]*W - low*np.maximum(W, 0.0) - high*np.minimum(W, 0.0)
            z = c.sum(axis = 0)
        else:
            c = np.maximum(a_in[:,None]*W, 0.0) if sign_split else a_in[:,None]*np.maximum(W, 0.0)
            z = c.sum(axis = 0) + (0.0 if (b is None or ignore_bias) else b)
        s = np.divide(R, z, out = np.zeros_like(R), where = z != 0)
        R = c @ s

    return R

def cosine(u, v):
    return (u*v).sum(axis = 1) / np.sqrt((u**2).sum(axis = 1) * (v**2).sum(axis = 1))

def agreement(a1, a2):
    """Agreement of two raw relevance arrays (samples, features), from the same samples."""

    r = cosine(a1 - a1.mean(axis = 1, keepdims = True), a2 - a2.mean(axis = 1, keepdims = True))
    diff = np.abs(a1 - a2).max()
    mean_map = [np.nanmean(a / np.abs(a).max(axis = 1, keepdims = True), axis = 0) for a in (a1, a2)]

    return {'max_abs_diff_relative': float(diff / max(np.abs(a1).max(), np.abs(a2).max())),
            'pearson_mean': float(np.nanmean(r)),
            'pearson_min': float(np.nanmin(r)),
            'cosine_mean': float(np.nanmean(cosine(a1, a2))),
            'pearson_mean_map': float(np.corrcoef(*mean_map)[0,1])}

if __name__ == "__main__":

    args = parse_args()

    keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    # On the A100s TensorFlow multiplies matrices in TF32 by default, which makes the Keras outputs differ from
    # PyTorch by about 1e-3 (job 3544505). Full float32 for the comparison.
    tf.config.experimental.enable_tensor_float_32_execution(False)

    # Required by innvestigate
    tf.compat.v1.disable_eager_execution()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    models_dir = join(args.out_dir, 'models')
    os.makedirs(models_dir, exist_ok = True)

    X, y = make_data(args.n_samples, args.n_features, args.n_informative, args.seed, args.positive_inputs)
    layers = make_layers(args.n_features)

    # The last samples are the test set, of the rest a random part is the validation set
    n_test = int(args.n_samples*TEST_FRAC)
    X_test, y_test = X[-n_test:], y[-n_test:]
    perm = np.random.default_rng(args.seed).permutation(args.n_samples - n_test)
    n_val = int((args.n_samples - n_test)*VAL_FRAC)
    x_v, y_v = X[perm[:n_val]], y[perm[:n_val]]
    x_t, y_t = X[perm[n_val:]], y[perm[n_val:]]

    samples = X_test[:args.n_explain]

    print("Train with Keras")
    keras_model = train_keras(x_t, y_t, x_v, y_v, layers, args, models_dir)
    keras_wo_softmax = innvestigate.model_wo_softmax(keras_model)

    print("Train with PyTorch")
    torch_model = train_torch(x_t, y_t, x_v, y_v, layers, args, models_dir)
    torch_wo_softmax = model_wo_softmax_torch(torch_model)

    print("Copy the Keras network into PyTorch")
    keras_torch_model = keras_to_torch(keras_model)
    keras_torch_wo_softmax = model_wo_softmax_torch(keras_torch_model)

    probs = {'keras'      : keras_model.predict(X_test),
             'torch'      : predict_torch(torch_model, X_test),
             'keras_torch': predict_torch(keras_torch_model, X_test)}

    results = {'positive_inputs': args.positive_inputs,
               'prediction_diff_keras_vs_keras_torch': float(np.abs(probs['keras'] - probs['keras_torch']).max()),
               'informative_share_chance': args.n_informative / args.n_features}
    for case, p in probs.items():
        results[f'accuracy_{case}'] = float((p.argmax(axis = 1) == y_test.argmax(axis = 1)).mean())

    print()
    print(f"Largest difference between the Keras and the PyTorch probabilities: {results['prediction_diff_keras_vs_keras_torch']:.2e}")
    for case in probs:
        print(f"{case:12s} accuracy {results[f'accuracy_{case}']:.3f}")

    numpy_layers = dense_layers(keras_wo_softmax)

    for key, name in METHOD_NAMES.items():
        method = dict(name = name, optParams = {})
        split = dict(name = name, optParams = {'sign_split': True})
        ignore_bias = name.endswith('_IB')

        print(f"\nAnalyze using {name}")
        a = {'keras'            : XAIKeras(keras_wo_softmax, method, 'classic', samples, RAW).quick_analyze()[0],
             'keras_torch'      : XAITorch(keras_torch_wo_softmax, method, 'classic', samples, RAW).quick_analyze()[0],
             'keras_torch_split': XAITorch(keras_torch_wo_softmax, split, 'classic', samples, RAW).quick_analyze()[0],
             'torch'            : XAITorch(torch_wo_softmax, method, 'classic', samples, RAW).quick_analyze()[0],
             'numpy_split'      : np.stack([numpy_a1b0(numpy_layers, s, True , ignore_bias) for s in samples]),
             'numpy_wplus'      : np.stack([numpy_a1b0(numpy_layers, s, False, ignore_bias) for s in samples])}
        a = {case: relevance.astype(np.float64) for case, relevance in a.items()}

        # Share of the relevance on the informative features. Relevance can be negative, hence abs.
        for case, relevance in a.items():
            share = np.abs(relevance[:,:args.n_informative]).sum(axis = 1) / np.abs(relevance).sum(axis = 1)
            results[f'informative_share_{key}_{case}'] = float(np.nanmean(share))
        print(f"informative share (chance {results['informative_share_chance']:.3f}): " +
              ", ".join(f"{case} {results[f'informative_share_{key}_{case}']:.3f}" for case in a))

        for c1, c2 in PAIRS:
            m = agreement(a[c1], a[c2])
            results[f'agreement_{key}_{c1}_vs_{c2}'] = m
            print(f"  {c1:17s} vs {c2:17s}: " + ", ".join(f"{k} = {v:.3g}" for k, v in m.items()))

    # LRP-A1B0-B of the ECCO scripts: Bounded rule in the first layer, A1B0 ignoring the bias in the others. innvestigate,
    # XAITorch's BoundedRule and the formula by hand, with innvestigate's default bounds and with the training range
    # (the signed X is often outside (-1, 1), where the rule's premise fails but the formula must still agree).
    data_range = (float(x_t.min()), float(x_t.max()))
    for key, input_layer_rule, bounds in (('b', 'Bounded', (-1.0, 1.0)), ('b_data', data_range, data_range)):
        method = dict(name = 'lrp.alpha_1_beta_0_IB', optParams = {'input_layer_rule': input_layer_rule})

        print(f"\nAnalyze using LRP-A1B0-B, bounds {bounds}")
        a = {'keras'       : XAIKeras(keras_wo_softmax, method, 'classic', samples, RAW).quick_analyze()[0],
             'keras_torch' : XAITorch(keras_torch_wo_softmax, method, 'classic', samples, RAW).quick_analyze()[0],
             'numpy_bounds': np.stack([numpy_a1b0(numpy_layers, s, True, True, bounds) for s in samples])}
        a = {case: relevance.astype(np.float64) for case, relevance in a.items()}

        for c1, c2 in (('keras', 'numpy_bounds'), ('keras_torch', 'numpy_bounds'), ('keras', 'keras_torch')):
            m = agreement(a[c1], a[c2])
            results[f'agreement_{key}_{c1}_vs_{c2}'] = m
            print(f"  {c1:17s} vs {c2:17s}: " + ", ".join(f"{k} = {v:.3g}" for k, v in m.items()))

    with open(join(args.out_dir, 'agreement_fake.json'), 'w') as f:
        json.dump(results, f, indent = 2)
