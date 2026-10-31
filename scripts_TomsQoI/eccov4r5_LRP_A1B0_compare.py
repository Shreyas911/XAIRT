"""
The same trained network explained by both XAI libraries: innvestigate (Keras) and Captum (PyTorch).

The other two scripts train a separate network per backend, so their relevance maps differ both
because of the training and because of the library. Here a network is trained once with Keras, its
weights are copied into a PyTorch model (keras_to_torch), and the methods of eccov4r5_LRP_A1B0_keras.py are run with each
library on exactly the same samples: LRP-A1B0-B of the notebook (Bounded rule in the first layer, A1B0 ignoring the bias in the
others) and LRP-A1B0. Any difference is due to the libraries.

Captum's own A1B0 differs from innvestigate's for inputs of both signs, such as the SST anomalies of the first layer: the positive
contributions are x_i (w_ij)+ in Captum and (x_i w_ij)+ in innvestigate (fake_data_LRP_A1B0_compare.py). So XAITorch runs here with
innvestigate's rules, sign_split for A1B0 and its BoundedRule for the first layer of A1B0-B, and agreement to float precision is
expected for both methods.

    python scripts_TomsQoI/eccov4r5_LRP_A1B0_compare.py                     # the saved Keras models of --saved-models-dir
    python scripts_TomsQoI/eccov4r5_LRP_A1B0_compare.py --source train --epochs 2 --lags 0    # trained, quick test

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
import innvestigate
import torch
import xarray as xr

# Append the src directory of the repository to sys.path, then import XAIRT
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from XAIRT import XAITorch, keras_to_torch, model_wo_softmax_torch

import eccov4r5_common as common
import eccov4r5_LRP_A1B0_keras as keras_script

METHODS = keras_script.METHODS

def explain_torch(model_wo_softmax, method, samples):
    """The method with innvestigate's A1B0 (sign_split), see the docstring."""
    method = dict(method, optParams = {**method['optParams'], 'sign_split': True})
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

    args = common.parse_args("compare", lambda p: common.add_model_args(p, ('saved', 'train')))

    keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    # On the A100s TensorFlow multiplies matrices in TF32 by default, which makes the Keras outputs differ from
    # PyTorch by about 1e-3 (job 3544505). Full float32 for the comparison.
    tf.config.experimental.enable_tensor_float_32_execution(False)

    # Required by innvestigate
    tf.compat.v1.disable_eager_execution()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ctx = common.make_context(args)
    X, oneHotCost, wetpoints = ctx.X, ctx.oneHotCost, ctx.wetpoints
    get_model = keras_script.make_get_model(args)

    results = {}
    maps = {}
    for lag in args.lags:

        print(f'Lag: {lag} days, for Theta')

        model = get_model(lag, ctx)
        keras_wo_softmax = innvestigate.model_wo_softmax(model)

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
