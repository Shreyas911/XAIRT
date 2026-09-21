"""
Plots of the simple correlation, LRP-A1B0-B and OI of the older runs, next to those of the runs with a smaller network,
converted from notebooks_TomsQoI/eccov4r5_results_viz.ipynb

    python scripts_TomsQoI/eccov4r5_results_viz.py --out-dir Results_old
    python scripts_TomsQoI/eccov4r5_results_viz.py --lags 0 60 --no-plots    # nothing is saved, to test the loading

Only plotting, nothing here needs Keras or Torch, so this script has a single version. It is the older of the two viz
notebooks, eccov4r5_results_viz_new.py is the one of the runs of the current scripts. A figure with 9 maps is made for each
lag of --lags: the correlation, LRP positive and negative, the same of the small network, OI positive and negative
and the same of the small network. The notebook made the figure for the lags 0, 60, 120, 180 and -60. The NetCDF files
of --results-dir are LRP_A1B0.nc, LRP_A1B0_small.nc, OI_v4r5_{pos,neg}.nc and OI_small_v4r5_{pos,neg}.nc. The maps of a
file that is not there are left out. Written to --out-dir: {lag}.png

The correlation is with the new anomalies, the correct ones, unlike in the notebook, which used the old ones.
"""

import os
from os.path import join

import eccov4r5_common as common
import eccov4r5_viz_common as viz

if __name__ == "__main__":

    args = common.parse_args("viz", viz.add_viz_args)
    os.makedirs(args.out_dir, exist_ok = True)

    X, y, wetpoints, XC, YC = viz.load_data(args)
    corr = viz.correlation_maps(X, y, args.lags, wetpoints)

    ds_lrp       = viz.open_results(args.results_dir, 'LRP_A1B0.nc')
    ds_lrp_small = viz.open_results(args.results_dir, 'LRP_A1B0_small.nc')
    ds_oi_pos       = viz.open_results(args.results_dir, 'OI_v4r5_pos.nc')
    ds_oi_pos_small = viz.open_results(args.results_dir, 'OI_small_v4r5_pos.nc')
    ds_oi_neg       = viz.open_results(args.results_dir, 'OI_v4r5_neg.nc')
    ds_oi_neg_small = viz.open_results(args.results_dir, 'OI_small_v4r5_neg.nc')

    if args.no_plots:
        raise SystemExit

    for lag in args.lags:
        name = common.lag_name(lag)
        lrp = lambda ds, cls: viz.field(ds, f"lrp_{name}_a1b0_b_{cls}_all")
        oi = lambda ds: viz.field(ds, f"OI_{name}")

        panels = [(corr[lag]                   , f"Simple correlation lag {lag} days"        , 'RdBu_r', -0.8, 0.8),
                  (lrp(ds_lrp, 'pos')          , f"LRP_A1B0 positive lag {lag} days"         , 'jet'   ,  0.0, 0.8),
                  (lrp(ds_lrp, 'neg')          , f"LRP_A1B0 negative lag {lag} days"         , 'jet'   ,  0.0, 0.8),
                  (lrp(ds_lrp_small, 'pos')    , f"LRP_A1B0 small positive lag {lag} days"   , 'jet'   ,  0.0, 0.8),
                  (lrp(ds_lrp_small, 'neg')    , f"LRP_A1B0 small negative lag {lag} days"   , 'jet'   ,  0.0, 0.8),
                  (oi(ds_oi_pos)               , f"OI positive lag {lag} days"               , 'RdBu_r', -1.0, 1.0),
                  (oi(ds_oi_neg)               , f"OI negative lag {lag} days"               , 'RdBu_r', -1.0, 1.0),
                  (oi(ds_oi_pos_small)         , f"OI positive small lag {lag} days"         , 'RdBu_r', -1.0, 1.0),
                  (oi(ds_oi_neg_small)         , f"OI negative small lag {lag} days"         , 'RdBu_r', -1.0, 1.0)]
        viz.plot_panels(panels, XC, YC, join(args.out_dir, f"{name}.png"), ncols = 3, figsize = (20,10))
