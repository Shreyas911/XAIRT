"""
The quick version of eccov4r5_LRP_A1B0_compare.py: no ECCO data, a fake classification problem with a known
answer, so it runs in a minute and can be used to check that the libraries agree.

Three cases, all LRP-A1B0 (alpha=1, beta=0 in every layer) and LRP-A1B0-IB (the same, ignoring the bias) on the same samples:

  keras        : network trained with Keras, explained with innvestigate
  torch        : network trained with PyTorch, explained with Captum
  keras_torch  : the network of 'keras', its weights copied into PyTorch (keras_to_torch), explained with Captum

'keras' vs 'keras_torch' is the same network, so any difference is due to innvestigate vs Captum.
'torch' is a different network, since it is trained separately, and is not expected to agree as well.

    python scripts_TomsQoI/fake_data_LRP_A1B0_compare.py --out-dir LRP_output_fake
    python scripts_TomsQoI/fake_data_LRP_A1B0_compare.py --epochs 5 --n-explain 20    # quick test

The data: X is standard normal and the class is the sign of a linear combination of only the first
--n-informative features plus noise, so the relevance should sit on those features. 'informative_share'
is the share of the (normalized) relevance that they get, to be compared with n_informative/n_features
for a network that ignores them. With --positive-inputs X is uniform in [0,1], and the score is centered.

Is agreement between 'keras' and 'keras_torch' expected? From reading the code of innvestigate 2.1.0 and Captum 0.9.0, not yet
from running it. Both put the raw bias in the denominator of A1B0, and the bias takes a share of the relevance in both, so the bias
is not a difference. The input is: innvestigate uses the positive weights on the positive part of the input and the negative
weights on the negative part, Captum only the positive weights on the input as it is. They are the same for inputs >= 0, which is
the case for the hidden layers, after a ReLU, but not for the first layer with the standard normal X. So:
  default          : the first layer differs, agreement below 1 is expected, for A1B0 and A1B0-IB
  --positive-inputs: every layer has inputs >= 0, agreement to float precision (Pearson about 1) is expected for both,
                     and if it is not there, the two libraries differ in something else

Hypothesis under test, and what each outcome would mean (see HANDOFF.md): keras vs keras_torch agree for inputs >= 0 and differ for
inputs of both signs. A difference with --positive-inputs would mean the libraries differ in something other than the input sign, and
agreement with signed inputs would mean the reading of the code is wrong. The bias is not expected to be a difference. Results of
XAITorch from before its rules were attached before every sample cannot be used, only the first sample got the rule.

Written to --out-dir: agreement_fake.json and the Keras model.
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

METHODS = [dict(name='lrp.alpha_1_beta_0'   , title = 'LRP-A1B0'   , optParams = {}),
           dict(name='lrp.alpha_1_beta_0_IB', title = 'LRP-A1B0-IB', optParams = {})]
NORMALIZE = {'bool_': True, 'kind': 'MaxAbs'}
LOSSES = [{'kind': 'categorical_crossentropy', 'weight': 1.0}]
BATCH_SIZE = 64
TEST_FRAC = 0.2
VAL_FRAC = 0.2

def parse_args():
    p = argparse.ArgumentParser(description="LRP-A1B0 with Keras, Torch and Keras-trained-Torch-explained, fake data.")
    p.add_argument("--out-dir", default="LRP_output_fake", help="model and results are written here")
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

def explain_keras(model_wo_softmax, method, samples):
    a, _ = XAIKeras(model_wo_softmax, method, 'classic', samples, NORMALIZE).quick_analyze()
    return a

def explain_torch(model_wo_softmax, method, samples):
    a, _ = XAITorch(model_wo_softmax, method, 'classic', samples, NORMALIZE).quick_analyze()
    return a

def method_key(method):
    """'LRP-A1B0-IB' -> 'a1b0_ib'"""
    return method['title'].lower().removeprefix('lrp-').replace('-', '_')

def cosine(u, v):
    return (u*v).sum(axis = 1) / np.sqrt((u**2).sum(axis = 1) * (v**2).sum(axis = 1))

def agreement(a1, a2):
    """Agreement of two relevance arrays (samples, features), from the same samples."""

    def pearson(u, v):
        return cosine(u - u.mean(axis = 1, keepdims = True), v - v.mean(axis = 1, keepdims = True))

    r = pearson(a1, a2)
    mean_map = [np.nanmean(a, axis = 0) for a in (a1, a2)]

    return {'pearson_mean': float(np.nanmean(r)),
            'pearson_min': float(np.nanmin(r)),
            'cosine_mean': float(np.nanmean(cosine(a1, a2))),
            'pearson_mean_map': float(np.corrcoef(*mean_map)[0,1])}

if __name__ == "__main__":

    args = parse_args()

    keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()

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

    results = {'prediction_diff_keras_vs_keras_torch': float(np.abs(probs['keras'] - probs['keras_torch']).max())}
    for case, p in probs.items():
        results[f'accuracy_{case}'] = float((p.argmax(axis = 1) == y_test.argmax(axis = 1)).mean())

    a = {}
    for method in METHODS:
        key = method_key(method)
        print(f"Analyze using {method['title']} with Keras")
        a[key] = {'keras': explain_keras(keras_wo_softmax, method, samples)}
        print(f"Analyze using {method['title']} with PyTorch")
        a[key]['torch'] = explain_torch(torch_wo_softmax, method, samples)
        print(f"Analyze using {method['title']} with PyTorch, on the Keras network")
        a[key]['keras_torch'] = explain_torch(keras_torch_wo_softmax, method, samples)

    pairs = (('keras', 'keras_torch'), ('keras', 'torch'), ('torch', 'keras_torch'))
    results['informative_share_chance'] = args.n_informative / args.n_features

    for key, relevance_of in a.items():
        relevance_of = {case: relevance.astype(np.float64) for case, relevance in relevance_of.items()}

        # Share of the relevance on the informative features. Relevance can be negative in some layers, hence abs.
        for case, relevance in relevance_of.items():
            share = np.abs(relevance[:,:args.n_informative]).sum(axis = 1) / np.abs(relevance).sum(axis = 1)
            results[f'informative_share_{key}_{case}'] = float(np.nanmean(share))

        for c1, c2 in pairs:
            results[f'agreement_{key}_{c1}_vs_{c2}'] = agreement(relevance_of[c1], relevance_of[c2])

    print()
    print(f"Largest difference between the Keras and the PyTorch probabilities: {results['prediction_diff_keras_vs_keras_torch']:.2e}")
    for case in probs:
        print(f"{case:12s} accuracy {results[f'accuracy_{case}']:.3f}")
    for method in METHODS:
        key = method_key(method)
        print(f"\n{method['title']}, informative share (chance {results['informative_share_chance']:.3f}): " +
              ", ".join(f"{case} {results[f'informative_share_{key}_{case}']:.3f}" for case in probs))
        for c1, c2 in pairs:
            print(f"  {c1} vs {c2}: " + ", ".join(f"{k} = {v:.4f}" for k, v in results[f'agreement_{key}_{c1}_vs_{c2}'].items()))

    with open(join(args.out_dir, 'agreement_fake.json'), 'w') as f:
        json.dump(results, f, indent = 2)
