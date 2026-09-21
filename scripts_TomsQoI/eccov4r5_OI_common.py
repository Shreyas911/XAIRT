"""
Shared by eccov4r5_OI_keras.py and eccov4r5_OI_torch.py, converted from
notebooks_TomsQoI/eccov4r5_classification-OI-pos-newAnomalies-shuffleVal-reweight-avgInit-savedNN.ipynb and
notebooks_TomsQoI/eccov4r5_classification-OI-neg-newAnomalies-shuffleVal-reweight-avgInit-savedNN.ipynb

The two notebooks only differ in the class that the optimal input (OI) is looked for, so it is the option
--kind here. For each lag a trained network is needed. The OI is then found by gradient descent on the
input, starting from the mean of the correctly predicted samples of that class (avgInit), on the loss of
the network output against the desired class. The data and the experiment are the ones of
eccov4r5_common.py. A backend script only has to say how to get a network and how to do the descent,
see run_oi().
"""

from os.path import join

import numpy as np
import xarray as xr

import eccov4r5_common as common

# The desired class of the network output
DESIRED = {'pos': np.array([[1.0, 0.0]], dtype = np.float32),
           'neg': np.array([[0.0, 1.0]], dtype = np.float32)}

def add_oi_args(parser, sources):
    common.add_model_args(parser, sources)
    parser.add_argument("--kind", choices = ['pos', 'neg', 'both'], default = 'both',
                        help="the class of the optimal input, the two notebooks are pos and neg")
    parser.add_argument("--eta", type = float, default = 0.9999, help="step of the gradient descent")
    parser.add_argument("--oi-iters", type = int, default = 10000, help="steps of the gradient descent")
    parser.add_argument("--print-freq", type = int, default = 200)

def run_oi(backend, args, get_model, predict_fn, optimal_input_fn):
    """
    The whole experiment, with the backend specific parts given as functions:

      get_model(lag, ctx) -> network, see make_get_model in the LRP script of the backend
      predict_fn(model, X) -> class probabilities (samples, 2)
      optimal_input_fn(model, desired_labels, init, eta, n_iters, print_freq) -> optimal input (wetpoints,)
    """

    ctx = common.make_context(args)

    kinds = ['pos', 'neg'] if args.kind == 'both' else [args.kind]
    nan_map = np.full(common.LLC_SHAPE, np.nan)
    fields = {kind: {} for kind in kinds}

    for lag in args.lags:

        print(f'Lag: {lag} days, for Theta')

        model = get_model(lag, ctx)
        idx_pos, idx_neg = common.correct_indices(predict_fn(model, ctx.X), ctx.oneHotCost, lag)

        for kind in kinds:
            idx_c = idx_pos if kind == 'pos' else idx_neg

            if len(idx_c) == 0:
                fields[kind][lag] = nan_map
                continue

            print(f"Optimal input for {kind} samples, starting from the mean of {len(idx_c)} correct samples")
            init = np.nanmean(ctx.X[idx_c], axis = 0)[np.newaxis,:]
            oi = optimal_input_fn(model, DESIRED[kind], init, args.eta, args.oi_iters, args.print_freq)
            fields[kind][lag] = common.to_llc(oi, ctx.wetpoints)

    for kind in kinds:
        # The same variable names as in the notebooks, e.g. OI_minus60, OI_0
        xr.Dataset({f"OI_{common.lag_name(lag)}": xr.DataArray(field) for lag, field in fields[kind].items()}).to_netcdf(
            join(args.out_dir, f'OI_v4r5_{kind}_newAnomalies_shuffleVal_reweight_avgInit_savedNN_{backend}.nc'))

        if not args.no_plots:
            common.plot_maps(fields[kind], ctx.XC, ctx.YC, f"OI_{kind} lag {{lag}} days",
                             join(args.out_dir, f'OI_{kind}_{backend}.png'),
                             cmap = 'RdBu_r', cmin = -1, cmax = 1)
