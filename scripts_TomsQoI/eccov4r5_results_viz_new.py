"""
Plots of the LRP-A1B0-B and OI results of the new anomalies runs, and of the simple correlation, converted from
notebooks_TomsQoI/eccov4r5_results_viz_new.ipynb

    python scripts_TomsQoI/eccov4r5_results_viz_new.py --out-dir Results
    python scripts_TomsQoI/eccov4r5_results_viz_new.py --lags 0 60 --no-plots    # nothing is saved, to test the loading

Only plotting, nothing here needs Keras or Torch, so this script has a single version. The plots are of the files
that eccov4r5_LRP_A1B0_keras.py / eccov4r5_LRP_A1B0_torch.py and eccov4r5_OI_keras.py / eccov4r5_OI_torch.py write,
give their names with --lrp-file, --oi-pos-file and --oi-neg-file (the defaults are the names of the notebook).
The plots of a file that is not there are left out. Written to --out-dir:
  {lag}.png                       the OI, LRP and the composites of one lag, for each lag
  oi_{pos,neg}.png, lrp_a1b0_ib_b_{pos,neg}.png, composite_{positive,negative}_samples.png, correlation.png
                                  one of these for all the lags
The notebook animation (imageio) was commented out and is not converted. In the notebook the two composite figures used a
stale index i instead of the lag, so every map was of the same lag, and the OI negative figure was titled OI-pos. Both are fixed.

The simple correlation needs the SST data. The notebook used the old anomalies for it while the results are of the
new anomalies, which are the correct ones, so the new ones are used here, unlike in the notebook.
"""

import os
from os.path import join

import cmocean

import eccov4r5_common as common
import eccov4r5_viz_common as viz

def add_args(p):
    viz.add_viz_args(p)
    p.add_argument("--lrp-file", default = "LRP_A1B0_IB_B_newAnomalies_shuffleVal_reweight.nc")
    p.add_argument("--oi-pos-file", default = "OI_v4r5_pos_newAnomalies_shuffleVal_reweight_avgInit_savedNN.nc")
    p.add_argument("--oi-neg-file", default = "OI_v4r5_neg_newAnomalies_shuffleVal_reweight_avgInit_savedNN.nc")

if __name__ == "__main__":

    args = common.parse_args("viz", add_args)

    os.makedirs(args.out_dir, exist_ok = True)

    X, y, wetpoints, XC, YC = viz.load_data(args)
    corr = viz.correlation_maps(X, y, args.lags, wetpoints)

    ds_lrp = viz.open_results(args.results_dir, args.lrp_file)
    ds_oi_pos = viz.open_results(args.results_dir, args.oi_pos_file)
    ds_oi_neg = viz.open_results(args.results_dir, args.oi_neg_file)

    if args.no_plots:
        raise SystemExit

    balance = cmocean.cm.balance

    # The results of each lag, {kind: {lag: field}}
    fields = {'oi_pos'  : {lag: viz.field(ds_oi_pos, f"OI_{common.lag_name(lag)}") for lag in args.lags},
              'oi_neg'  : {lag: viz.field(ds_oi_neg, f"OI_{common.lag_name(lag)}") for lag in args.lags},
              'lrp_pos' : {lag: viz.field(ds_lrp, f"lrp_{common.lag_name(lag)}_a1b0_b_pos_all") for lag in args.lags},
              'lrp_neg' : {lag: viz.field(ds_lrp, f"lrp_{common.lag_name(lag)}_a1b0_b_neg_all") for lag in args.lags},
              'comp_pos': {lag: viz.field(ds_lrp, f"lrp_{common.lag_name(lag)}_comp_pos_all") for lag in args.lags},
              'comp_neg': {lag: viz.field(ds_lrp, f"lrp_{common.lag_name(lag)}_comp_neg_all") for lag in args.lags}}

    # One figure for each lag
    for lag in args.lags:
        panels = [(fields['oi_pos'][lag]  , "Optimal Input positive"    , 'RdBu_r', -1.0, 1.0),
                  (fields['oi_neg'][lag]  , "Optimal Input negative"    , 'RdBu_r', -1.0, 1.0),
                  (fields['lrp_pos'][lag] , "LRP positive"              , 'jet'   ,  0.0, 0.3),
                  (fields['lrp_neg'][lag] , "LRP negative"              , 'jet'   ,  0.0, 0.3),
                  (fields['comp_pos'][lag], "Composite of positive samples", 'RdBu_r', -0.8, 0.8),
                  (fields['comp_neg'][lag], "Composite of negative samples", 'RdBu_r', -0.8, 0.8)]
        viz.plot_panels(panels, XC, YC, join(args.out_dir, f"{common.lag_name(lag)}.png"), ncols = 2, figsize = (15,10))

    # One figure for each result, with all the lags
    for filename, panels in (("oi_pos.png"                     , viz.lag_grid(fields['oi_pos']  , "OI-pos lag {lag} days"             , 'RdBu_r', -1.0, 1.0)),
                             ("oi_neg.png"                     , viz.lag_grid(fields['oi_neg']  , "OI-neg lag {lag} days"             , 'RdBu_r', -1.0, 1.0)),
                             ("lrp_a1b0_ib_b_pos.png"          , viz.lag_grid(fields['lrp_pos'] , "LRP-A1B0-IB-B lag {lag} days"      , 'jet'   ,  0.0, 0.3)),
                             ("lrp_a1b0_ib_b_neg.png"          , viz.lag_grid(fields['lrp_neg'] , "LRP-A1B0-IB-B lag {lag} days"      , 'jet'   ,  0.0, 0.3)),
                             ("composite_positive_samples.png" , viz.lag_grid(fields['comp_pos'], "Comp. observations (pos_all) lag {lag} days", balance, -1, 1)),
                             ("composite_negative_samples.png" , viz.lag_grid(fields['comp_neg'], "Comp. observations (neg_all) lag {lag} days", balance, -1, 1)),
                             ("correlation.png"                , viz.lag_grid(corr              , "Correlation lag {lag} days"        , balance, -1, 1))):
        viz.plot_panels(panels, XC, YC, join(args.out_dir, filename), ncols = 3, figsize = (20,10))
