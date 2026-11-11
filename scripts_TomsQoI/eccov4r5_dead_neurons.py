"""
Why do some trained networks give no LRP relevance, and no OI movement, for a whole class? (HANDOFF.md, the 500-epoch Keras networks
of job 3544560 at lags -30 and 0 (neg) and 120 (both classes).) Suspected: ReLU neurons that are never active (dead), so that a
class logit does not depend on the input at all near the data.

Two modes, both writing dead_neurons.json to --out-dir and printing one line per network:

  inspect: networks that exist, --inspect LABEL=DIR ..., DIR holding model{lag}_noL1.h5 (Keras) or model{lag}_noL1.pt (PyTorch)
      python scripts_TomsQoI/eccov4r5_dead_neurons.py --inspect keras500=<run>/LRP_keras/models torch500=<run>/LRP_torch/models

  train: networks trained here as the LRP scripts do (make_train_fn of each backend), for every --backends x --seeds x --epochs-list
  x --lags, then inspected. Answers whether the dead networks are a property of Keras' training or of the seed.
      python scripts_TomsQoI/eccov4r5_dead_neurons.py --backends keras torch --seeds 42 43 44 --epochs-list 50 500 --lags 0 120

The Keras LRP script trains in TF1 graph mode (it disables eager execution for innvestigate before training) and with TF32 on.
This script does the same by default; --keras-eager and --no-tf32 train Keras otherwise, to test those as causes.

Per network, on all samples (X, lag aligned as for training) and on the correctly predicted samples of each class:
  active_share      share of the samples on which each hidden neuron is active, per layer
  n_dead            neurons active on no sample, per layer
  constant_share    per class, share of its correctly predicted samples whose logit has no active path from the input (its
                    gradient w.r.t. the input is zero: OI cannot move, every LRP rule gives zero relevance)
  blocked_share     per class, share in which A1B0 relevance cannot reach the inputs (mechanism() of eccov4r5_negative_class_LRP.py)
  bias, w_norm      per neuron of the first hidden layer: its bias and the norm of its input weights
"""

import json
import os
import sys
from os.path import exists, join
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import eccov4r5_common as common

def add_args(p):
    p.add_argument("--inspect", nargs = "+", default = [], metavar = "LABEL=DIR",
                   help="inspect the networks model{lag}_noL1.h5 / .pt of DIR")
    p.add_argument("--backends", nargs = "+", choices = ('keras', 'torch'), default = [],
                   help="train networks with these backends (train mode)")
    p.add_argument("--seeds", type = int, nargs = "+", default = [42])
    p.add_argument("--epochs-list", type = int, nargs = "+", default = [500])
    p.add_argument("--keras-eager", action = "store_true", help="train Keras in eager mode (the LRP script uses graph mode)")
    p.add_argument("--no-tf32", action = "store_true", help="train Keras without TF32 (the LRP script uses TF32)")

args = common.parse_args("dead_neurons", add_args)

import tensorflow as tf
import tensorflow.keras as keras
if not args.keras_eager:
    tf.compat.v1.disable_eager_execution()    # as eccov4r5_LRP_A1B0_keras.py
if args.no_tf32:
    tf.config.experimental.enable_tensor_float_32_execution(False)
import torch

from XAIRT import metricF1
from eccov4r5_negative_class_LRP import mechanism

def weights_of(model):
    """[(W (in, out), b)] of a Keras model or a Torch nn.Sequential, the last layer included (before the softmax)."""
    if isinstance(model, torch.nn.Module):
        return [(m.weight.detach().cpu().numpy().T.astype(np.float64), m.bias.detach().cpu().numpy().astype(np.float64))
                for m in model.modules() if isinstance(m, torch.nn.Linear)]
    return [tuple(w.astype(np.float64) for w in layer.get_weights()) for layer in model.layers
            if isinstance(layer, keras.layers.Dense)]

def load(dirname, lag, ctx):
    h5, pt = join(dirname, f'model{lag}_noL1.h5'), join(dirname, f'model{lag}_noL1.pt')
    if exists(h5):
        return keras.models.load_model(h5, custom_objects = {'metricF1': metricF1})
    if exists(pt):
        from XAIRT import TrainTorchFullyConnectedNN
        return TrainTorchFullyConnectedNN.load_saved_model(pt, ctx.layers)
    raise FileNotFoundError(f"neither {h5} nor {pt}")

def softmax(z):
    e = np.exp(z - z.max(axis = 1, keepdims = True))
    return e / e.sum(axis = 1, keepdims = True)

def inspect(weights, ctx, lag):
    X, _ = common.lag_align(ctx.X, ctx.oneHotCost, lag)
    hs = [X.astype(np.float64)]
    for W, b in weights[:-1]:
        hs.append(np.maximum(hs[-1] @ W + b, 0.0))
    # The correctly predicted samples of each class, as the LRP and OI scripts select them (probabilities of all of X)
    idx_pos, idx_neg = common.correct_indices(softmax(full_logits(weights, ctx.X)), ctx.oneHotCost, lag)

    r = {'active_share': [[float(s) for s in (h > 0).mean(axis = 0)] for h in hs[1:]],
         'n_dead': [int(((h > 0).sum(axis = 0) == 0).sum()) for h in hs[1:]],
         'bias_h1': [float(v) for v in weights[0][1]],
         'w_norm_h1': [float(v) for v in np.linalg.norm(weights[0][0], axis = 0)]}

    for cls, c, idx in (('pos', 0, idx_pos), ('neg', 1, idx_neg)):
        Xc = ctx.X[idx].astype(np.float64)
        hc = [Xc]
        for W, b in weights[:-1]:
            hc.append(np.maximum(hc[-1] @ W + b, 0.0))
        # Neurons with an active path to logit c, from the top down; the logit is constant if none of the first layer has one
        reach = (hc[-1] > 0) & (weights[-1][0][:, c] != 0)
        for k in range(len(weights) - 2, 0, -1):
            reach = (hc[k] > 0) & ((weights[k][0] != 0)[None] & reach[:, None, :]).any(axis = 2)
        constant = reach.sum(axis = 1) == 0
        _, _, _, _, reached = mechanism(weights, Xc, c)
        blocked = np.any(np.stack(reached) == 0, axis = 0)
        r[cls] = {'n': len(idx), 'constant_share': float(constant.mean()) if len(idx) else float('nan'),
                  'blocked_share': float(blocked.mean()) if len(idx) else float('nan')}
    return r

def full_logits(weights, X):
    h = X.astype(np.float64)
    for W, b in weights[:-1]:
        h = np.maximum(h @ W + b, 0.0)
    return h @ weights[-1][0] + weights[-1][1]

def report(key, r):
    print(f"{key}: dead {r['n_dead']}, "
          f"pos n {r['pos']['n']} constant {r['pos']['constant_share']:.3f} blocked {r['pos']['blocked_share']:.3f}, "
          f"neg n {r['neg']['n']} constant {r['neg']['constant_share']:.3f} blocked {r['neg']['blocked_share']:.3f}, "
          f"h1 active share per neuron {[round(s, 2) for s in r['active_share'][0]]}, "
          f"h2 {[round(s, 2) for s in r['active_share'][1]]}, h1 bias {[round(v, 3) for v in r['bias_h1']]}", flush = True)

if __name__ == "__main__":

    ctx = common.make_context(args)
    results = {}

    for item in args.inspect:
        label, dirname = item.split("=", 1)
        for lag in args.lags:
            key = f"{label} lag {lag}"
            results[key] = inspect(weights_of(load(dirname, lag, ctx)), ctx, lag)
            report(key, results[key])

    for backend in args.backends:
        if backend == 'keras':
            import eccov4r5_LRP_A1B0_keras as backend_script
        else:
            import eccov4r5_LRP_A1B0_torch as backend_script
        for seed in args.seeds:
            for epochs in args.epochs_list:
                run_args = type(args)(**{**vars(args), 'seed': seed, 'epochs': epochs})
                train = backend_script.make_train_fn(run_args)
                models_dir = join(args.out_dir, 'models', f'{backend}_seed{seed}_epochs{epochs}')
                os.makedirs(models_dir, exist_ok = True)
                for lag in args.lags:
                    if backend == 'keras':
                        keras.utils.set_random_seed(seed)
                    else:
                        torch.manual_seed(seed)
                    x_t, x_v, y_t, y_v = common.split_lag(ctx.X[:ctx.idx], ctx.oneHotCost[:ctx.idx], lag, common.VAL_SPLIT_FRAC)
                    model, _ = train(x_t, y_t, x_v, y_v, lag, ctx.layers, ctx.class_weight, models_dir)
                    key = f"{backend} seed {seed} epochs {epochs} lag {lag}"
                    results[key] = inspect(weights_of(model), ctx, lag)
                    report(key, results[key])
                    with open(join(args.out_dir, 'dead_neurons.json'), 'w') as f:
                        json.dump(results, f, indent = 1)

    with open(join(args.out_dir, 'dead_neurons.json'), 'w') as f:
        json.dump(results, f, indent = 1)
