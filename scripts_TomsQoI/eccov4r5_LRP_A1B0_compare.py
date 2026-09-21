"""
The same trained network explained by both XAI libraries: innvestigate (Keras) and Captum (PyTorch).

The other two scripts train a separate network per backend, so their relevance maps differ both
because of the training and because of the library. Here a network is trained once with Keras, its
weights are copied into a PyTorch model (keras_to_torch), and LRP-A1B0, alpha=1 and beta=0 in every
layer, and LRP-A1B0-IB, the same ignoring the bias, are run with each library on exactly the same samples.
Any difference is due to the libraries.

Is agreement expected? From reading the code of innvestigate 2.1.0 and Captum 0.9.0, not yet from running it: both put the raw
bias in the denominator of A1B0 and let it keep a share of the relevance, so the bias is not a difference. The input is:
innvestigate uses the positive weights on the positive part of the input and the negative weights on the negative part, Captum only
the positive weights on the input as it is. They are the same for inputs >= 0, the hidden layers after a ReLU, but not for the first
layer here, since the SST anomalies have both signs. So agreement below 1 is expected for both methods, and it is the first layer
that differs. The notebook uses the Bounded rule for the first layer, which Captum does not have.

Hypothesis under test, and what each outcome would mean (see HANDOFF.md): keras vs keras_torch agree for inputs >= 0 and differ for
inputs of both signs. A difference with --positive-inputs would mean the libraries differ in something other than the input sign, and
agreement with signed inputs would mean the reading of the code is wrong. The bias is not expected to be a difference. Results of
XAITorch from before its rules were attached before every sample cannot be used, only the first sample got the rule.

    python scripts_TomsQoI/eccov4r5_LRP_A1B0_compare.py --out-dir LRP_output_compare
    python scripts_TomsQoI/eccov4r5_LRP_A1B0_compare.py --epochs 2 --lags 0    # quick test

Written to --out-dir: agreement_compare.json (the numbers below, for each class and method), the normalized mean relevance
maps of both libraries in a NetCDF file, and the Keras models.

Per class (pos, neg) and lag it reports
  max_abs_prediction_diff : largest difference between the Keras and the PyTorch class probabilities,
                            about 1e-6 if the weights were copied correctly
  pearson_mean, pearson_min : Pearson correlation of the two relevance vectors of a sample, mean and
                            minimum over the samples
  cosine_mean             : the same with the cosine similarity
  pearson_mean_map        : Pearson correlation of the two normalized mean maps, the ones that are plotted
"""

import json
import os
import sys
from os.path import join
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import torch
import xarray as xr

# Append the src directory of the repository to sys.path, then import XAIRT
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from XAIRT import XAITorch, keras_to_torch, model_wo_softmax_torch

import eccov4r5_common as common
import eccov4r5_LRP_A1B0_keras as keras_script

METHODS = [dict(name='lrp.alpha_1_beta_0'   , title = 'LRP-A1B0'   , optParams = {}),
           dict(name='lrp.alpha_1_beta_0_IB', title = 'LRP-A1B0-IB', optParams = {})]

def explain_torch(model_wo_softmax, method, samples):
    Xplain = XAITorch(model_wo_softmax, method, 'classic', samples, common.NORMALIZE)
    a, _ = Xplain.quick_analyze()
    return a

def agreement(a_keras, a_torch):
    """Agreement of two relevance arrays (samples, wetpoints), from the same samples."""

    a_keras = a_keras.astype(np.float64)
    a_torch = a_torch.astype(np.float64)

    def cosine(u, v):
        return (u*v).sum(axis = 1) / np.sqrt((u**2).sum(axis = 1) * (v**2).sum(axis = 1))

    def pearson(u, v):
        return cosine(u - u.mean(axis = 1, keepdims = True), v - v.mean(axis = 1, keepdims = True))

    # A sample with NaN relevance (all zero before the normalization) is left out by the nan functions
    r = pearson(a_keras, a_torch)
    c = cosine(a_keras, a_torch)

    map_keras = common.normalized_mean_relevance(a_keras)
    map_torch = common.normalized_mean_relevance(a_torch)
    finite = np.isfinite(map_keras) & np.isfinite(map_torch)

    return {'n': int(a_keras.shape[0]),
            'pearson_mean': float(np.nanmean(r)),
            'pearson_min': float(np.nanmin(r)),
            'cosine_mean': float(np.nanmean(c)),
            'pearson_mean_map': float(np.corrcoef(map_keras[finite], map_torch[finite])[0,1])}, map_keras, map_torch

if __name__ == "__main__":

    args = common.parse_args("compare")

    keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()

    # Required by innvestigate
    tf.compat.v1.disable_eager_execution()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(args.out_dir, exist_ok = True)
    models_dir = join(args.out_dir, 'models')
    os.makedirs(models_dir, exist_ok = True)

    X, oneHotCost, wetpoints, XC, YC = common.load_data(args.r4_dir, args.r5_dir)
    layers = common.make_layers(X.shape[1])
    class_weight = common.class_weights(oneHotCost)
    train = keras_script.make_train_fn(args)

    idx = int(X.shape[0]*(1-common.TEST_SPLIT_FRAC))

    results = {}
    maps = {}
    for lag in args.lags:

        print(f'Lag: {lag} days, for Theta')

        x_t, x_v, y_t, y_v = common.split_lag(X[:idx], oneHotCost[:idx], lag, common.VAL_SPLIT_FRAC)
        model, keras_wo_softmax = train(x_t, y_t, x_v, y_v, lag, layers, class_weight, models_dir)

        torch_model = keras_to_torch(model)
        torch_wo_softmax = model_wo_softmax_torch(torch_model)

        probs_keras = model.predict(X)
        with torch.no_grad():
            probs_torch = torch_model(torch.from_numpy(X).float()).numpy()

        result = {'max_abs_prediction_diff': float(np.abs(probs_keras - probs_torch).max())}
        print(f"Largest difference between the Keras and the PyTorch probabilities: {result['max_abs_prediction_diff']:.2e}")

        idx_pos, idx_neg = common.correct_indices(probs_keras, oneHotCost, lag)

        for cls, idx_c in (('pos', idx_pos), ('neg', idx_neg)):

            if len(idx_c) == 0:
                continue

            result[cls] = {}
            for method in METHODS:
                key = common.title_key(method['title'])

                print(f"Analyze using {method['title']} for {cls} samples")
                a_keras = keras_script.explain(keras_wo_softmax, method, X[idx_c])
                a_torch = explain_torch(torch_wo_softmax, method, X[idx_c])

                result[cls][key], map_keras, map_torch = agreement(a_keras, a_torch)
                maps[f"lrp_{common.lag_name(lag)}_{key}_{cls}_keras"] = common.to_llc(map_keras, wetpoints)
                maps[f"lrp_{common.lag_name(lag)}_{key}_{cls}_torch"] = common.to_llc(map_torch, wetpoints)

                print(f"  {cls}, {method['title']}: " + ", ".join(f"{k} = {v:.4f}" for k, v in result[cls][key].items() if k != 'n'))

        results[str(lag)] = result

    with open(join(args.out_dir, 'agreement_compare.json'), 'w') as f:
        json.dump(results, f, indent = 2)

    xr.Dataset({name: xr.DataArray(field) for name, field in maps.items()}).to_netcdf(
        join(args.out_dir, 'LRP_A1B0_newAnomalies_shuffleVal_reweight_compare.nc'))
