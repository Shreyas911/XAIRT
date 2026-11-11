"""
Why does LRP give no relevance to many samples of the negative class? See "Open research question" in HANDOFF.md.

    python scripts_TomsQoI/eccov4r5_negative_class_LRP.py                    # the saved Keras models, all 9 lags
    python scripts_TomsQoI/eccov4r5_negative_class_LRP.py --lags 0 60 --n-lrp 200

The suspected mechanism: A1B0 distributes only the positive contributions (h_i w_ij)+ into each neuron j. If a neuron that
carries relevance has none (it is active only through its bias), its relevance goes nowhere below it (with the bias kept, all of
it goes to the bias). If that happens for every relevance-carrying neuron of some layer, every input gets zero relevance, for any
rule of the first layer, Bounded included. The first version only checked the last layer (the logit); on the saved models
(job 3545869) that never happened although LRP gave all-zero relevance, so the check now goes through every hidden layer.

For each lag and class, on the correctly predicted samples that the LRP scripts explain:
  1. mechanism, all samples, a forward pass: the share with no positive contribution into the logit, the share in which A1B0
     relevance cannot reach the inputs (a hidden layer with no neuron that carries it, blocked_share), the bias share of the
     logit, the number of active neurons in each hidden layer
  2. LRP on the first --n-lrp samples: the share of all-zero relevance for LRP-A1B0-B and LRP-A1B0, expected to match 1
  3. a remedy: LRP-A1B0-B of the logit difference z_c - z_other (the last layer replaced by w_c - w_other, b_c - b_other, the
     same predictions), its share of all-zero relevance, and the correlation of its mean map with the one of LRP-A1B0-B
And the share of NaN in each map of the published file (--published).

Written to --out-dir: negative_class.json.
"""

import json
import os
import sys
from os.path import exists, join
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import innvestigate
import xarray as xr

# Append the src directory of the repository to sys.path, then import XAIRT
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from XAIRT import XAIKeras

import eccov4r5_common as common
import eccov4r5_LRP_A1B0_keras as keras_script

RAW = {'bool_': False}
METHODS = {'a1b0_b': dict(name='lrp.alpha_1_beta_0_IB', optParams = {'input_layer_rule': 'Bounded'}),
           'a1b0'  : dict(name='lrp.alpha_1_beta_0'   , optParams = {})}

def add_args(p):
    common.add_model_args(p, ('saved', 'saved-torch', 'train'))
    p.add_argument("--n-lrp", type = int, default = 500, help="samples per class explained with LRP (the mechanism uses all)")
    p.add_argument("--published", default = str(Path(__file__).resolve().parents[1] /
                                                "examples_TomsQoI/LRP_output_forHelen/LRP_A1B0_IB_B_newAnomalies_shuffleVal_reweight.nc"),
                   help="the NetCDF file of the published LRP maps")

def dense_weights(model):
    """(W, b) of the Dense layers of the model, the last one included (before its softmax)."""
    return [layer.get_weights() for layer in model.layers if isinstance(layer, keras.layers.Dense)]

def mechanism(weights, X, c):
    """
    Per sample: the sum of the positive contributions into logit c, the logit, its bias, the active neurons of each hidden layer,
    and the layers that A1B0 relevance reaches. A1B0 passes relevance from neuron j down to neuron i only through a positive
    contribution h_i w_ij > 0, so a neuron carries relevance if it has a positive contribution into a neuron above that carries
    it. reached[k] is the number of neurons of hidden layer k+1 that carry relevance; if a hidden layer has none, every input
    gets zero relevance, whatever the rule of the first layer.
    """
    hs, active = [X.astype(np.float64)], []
    for W, b in weights[:-1]:
        hs.append(np.maximum(hs[-1] @ W + b, 0.0))
        active.append((hs[-1] > 0).sum(axis = 1))
    W, b = weights[-1]
    h = hs[-1]
    pos = np.maximum(h * W[:,c], 0.0).sum(axis = 1)
    logit = h @ W[:,c] + b[c]

    carry = h * W[:,c] > 0    # (samples, neurons of the last hidden layer)
    reached = [carry.sum(axis = 1)]
    for k in range(len(weights) - 2, 0, -1):    # the hidden layers below, down to the first hidden layer
        W_k = weights[k][0]
        carry = ((hs[k][:,:,None] * W_k[None]) > 0) & carry[:,None,:]    # (samples, i, j)
        carry = carry.any(axis = 2)
        reached.insert(0, carry.sum(axis = 1))
    return pos, logit, float(b[c]), active, reached

def difference_model(weights, c):
    """
    A copy of the network with its last layer replaced by one output, the logit difference z_c - z_other. New layers, since
    innvestigate does not take layers that are used in two models. Zero initializers: the weights are set right after, and a
    random initializer fails in graph mode with op determinism on ("Random ops require a seed").
    """
    zeros = dict(kernel_initializer = 'zeros', bias_initializer = 'zeros')
    inp = keras.Input(shape = (weights[0][0].shape[0],))
    x, layers = inp, []
    for W, b in weights[:-1]:
        layers.append((keras.layers.Dense(W.shape[1], activation = 'relu', **zeros), [W, b]))
        x = layers[-1][0](x)
    W, b = weights[-1]
    layers.append((keras.layers.Dense(1, activation = None, **zeros), [(W[:,c] - W[:,1-c])[:,None], np.array([b[c] - b[1-c]])]))
    out = layers[-1][0](x)
    for layer, w in layers:
        layer.set_weights(w)
    return keras.Model(inp, out)

def all_zero_share(a):
    return float((np.abs(a).sum(axis = 1) == 0).mean())

def corr(u, v):
    m = np.isfinite(u) & np.isfinite(v)
    return float(np.corrcoef(u[m], v[m])[0,1]) if m.sum() > 1 else float('nan')

if __name__ == "__main__":

    args = common.parse_args("keras", add_args)

    keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    tf.compat.v1.disable_eager_execution()    # innvestigate

    ctx = common.make_context(args)
    get_model = keras_script.make_get_model(args)
    results = {}

    if exists(args.published):
        # Share of the wet points (the inputs of the network) that are NaN in each map; land and the masked latitudes are NaN anyway
        ds = xr.open_dataset(args.published)
        results['published_nan_share_of_wetpoints'] = {}
        print("Share of the wet points that are NaN in the published maps:")
        for name in ds.data_vars:
            values = ds[name].values
            if values.shape[-3:] != common.LLC_SHAPE:
                continue
            share = float(np.isnan(values[..., ctx.wetpoints[0], ctx.wetpoints[1], ctx.wetpoints[2]]).mean())
            results['published_nan_share_of_wetpoints'][name] = share
            print(f"  {name} {values.shape}: {share:.3f}")
    else:
        print(f"Not found, skipped: {args.published}")

    for lag in args.lags:

        model = get_model(lag, ctx)
        keras_wo_softmax = innvestigate.model_wo_softmax(model)
        weights = dense_weights(keras_wo_softmax)
        idx_pos, idx_neg = common.correct_indices(model.predict(ctx.X, batch_size = 1024), ctx.oneHotCost, lag)
        results[str(lag)] = {}

        for cls, c, idx_c in (('pos', 0, idx_pos), ('neg', 1, idx_neg)):
            if len(idx_c) == 0:
                continue
            X_c = ctx.X[idx_c]
            pos, logit, bias, active, reached = mechanism(weights, X_c, c)
            blocked = np.any(np.stack(reached) == 0, axis = 0)
            r = {'n': len(idx_c),
                 'no_positive_contribution_share': float((pos == 0).mean()),
                 'blocked_share': float(blocked.mean()),
                 'blocked_share_by_layer': [float((n == 0).mean()) for n in reached],
                 'bias': bias,
                 'bias_share_of_logit_median': float(np.median(bias / logit)),
                 'active_neurons_median': [float(np.median(n)) for n in active],
                 'relevance_neurons_median': [float(np.median(n)) for n in reached]}

            sub = X_c[:args.n_lrp]
            r['n_lrp'] = len(sub)
            r['no_positive_contribution_share_lrp_samples'] = float((pos[:args.n_lrp] == 0).mean())
            r['blocked_share_lrp_samples'] = float(blocked[:args.n_lrp].mean())
            maps = {}
            for key, method in METHODS.items():
                a = XAIKeras(keras_wo_softmax, method, 'classic', sub, RAW).quick_analyze()[0].astype(np.float64)
                r[f'all_zero_share_{key}'] = all_zero_share(a)
                maps[key] = common.normalized_mean_relevance(a)

            a = XAIKeras(difference_model(weights, c), METHODS['a1b0_b'], 'classic', sub, RAW).quick_analyze()[0].astype(np.float64)
            r['all_zero_share_a1b0_b_difference'] = all_zero_share(a)
            r['corr_mean_map_a1b0_b_vs_difference'] = corr(maps['a1b0_b'], common.normalized_mean_relevance(a))

            results[str(lag)][cls] = r
            print(f"lag {lag:4d} {cls}: " + ", ".join(f"{k} = {v:.3f}" if isinstance(v, float) else f"{k} = {v}" for k, v in r.items()))

    os.makedirs(args.out_dir, exist_ok = True)
    with open(join(args.out_dir, 'negative_class.json'), 'w') as f:
        json.dump(results, f, indent = 2)
