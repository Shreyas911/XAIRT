"""
ECCOv4r5 SST data: builds SST_all.nc, and the statistics and the plots of the new anomalies, converted from
notebooks_TomsQoI/eccov4r5_dataReader-newAnomalies.ipynb

    python scripts_TomsQoI/eccov4r5_dataReader.py --build-sst              # once, writes SST_all.nc to --r5-dir
    python scripts_TomsQoI/eccov4r5_dataReader.py --out-dir Results_data
    python scripts_TomsQoI/eccov4r5_dataReader.py --no-plots               # only the statistics

Only data, nothing here needs Keras or Torch, so this script has a single version.

--build-sst reads the daily mds SST files of --r5-dir (SST_day_mean and SST_day_mean_ext_2020_2023_Jun, with GRID/), puts them
in tiles and writes SST_all.nc, the input of all the other ECCOv4r5 scripts. In the notebook this was commented out. It stops
before overwriting a file that is there, use --force to do it.

The rest is on the new anomalies, the ones of the LRP and OI scripts, and prints the statistics that were used to justify the
LRP-Bounded rule for the first layer (the range of the input, the share of it within +-1, +-2 and +-3), and writes to --out-dir:
  statistics_maps.png       max, min and standard deviation of the anomalies at each grid cell
  std_histogram.png         histogram of that standard deviation
  anomaly_maps.png          the anomalies of the first day, the day 182, a year before the end and the last day
  convolution_effect.png    the QoI before and after the 61-day running mean
  correlation.png           correlation of the anomalies with the QoI for --lags
  correlation_long_lags.png the same for --long-lags
The notebook also plotted the SST itself (first day, mean, the objective function point) and compared the new anomalies with the
old ones, which are not converted.
"""

import os
from os.path import exists, join

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import xarray as xr

import eccov4r5_common as common
import eccov4r5_viz_common as viz

N_DAYS = 11323    # 1992-01-01 to 2022-12-31, the files have up to 2023-06

def add_args(p):
    p.set_defaults(out_dir = "Results_data")
    p.add_argument("--build-sst", action = "store_true", help="write SST_all.nc first")
    p.add_argument("--force", action = "store_true", help="overwrite SST_all.nc")
    p.add_argument("--long-lags", type = int, nargs = "+", default = [-1500,-1000,-500,-365,365,500,1000,1500,2000],
                   help="lags in days of the second correlation figure")

def build_sst(r5_dir, force):

    # Only needed here
    import xmitgcm
    import ecco_v4_py as ecco

    path = join(r5_dir, 'SST_all.nc')
    if exists(path) and not force:
        raise SystemExit(f"{path} is there, --force to overwrite it")

    gridDir = join(r5_dir, 'GRID')
    thetaDir = join(r5_dir, 'SST_day_mean')
    thetaDir_ext = join(r5_dir, 'SST_day_mean_ext_2020_2023_Jun')

    # SSH has to be kept because someone used the SSH metadata for SST,
    # It's not a bug in this code but a hack to handle an existing bug.
    temp = xmitgcm.open_mdsdataset(data_dir = thetaDir,
                                   grid_dir = gridDir,
                                   extra_variables = dict(SSH = dict(dims=['k','j','i'],
                                                                     attrs = dict(standard_name="SST",
                                                                                  long_name="Sea Surface Temperature",
                                                                                  units="degC"))))
    temp["SST"] = temp["SSH"]
    temp = temp.drop(["SSH"])

    temp_ext = xmitgcm.open_mdsdataset(data_dir = thetaDir_ext,
                                       grid_dir = gridDir,
                                       extra_variables = dict(SST = dict(dims=['k','j','i'],
                                                                         attrs = dict(standard_name="SST",
                                                                                      long_name="Sea Surface Temperature",
                                                                                      units="degC"))))

    ds_r5 = xr.concat([temp, temp_ext], "time")

    # In chunks of 1000 days, to have less in memory
    SST = np.concatenate([ecco.llc_compact_to_tiles(ds_r5['SST'][start:min(start+1000, N_DAYS)])
                          for start in range(0, N_DAYS, 1000)], axis = 0)

    xr.Dataset({'SST': xr.DataArray(SST)}).to_netcdf(path)
    print(f"Wrote {path}, {SST.shape}")

if __name__ == "__main__":

    args = common.parse_args("dataReader", add_args)

    if args.build_sst:
        build_sst(args.r5_dir, args.force)

    X_full, y_full, wetpoints, XC, YC = common.load_anomalies(args.r4_dir, args.r5_dir)
    X, y, oneHotCost = common.make_qoi(X_full, y_full)

    print(f"Samples of the classes pos and neg: {int(oneHotCost[:,0].sum())}, {int(oneHotCost[:,1].sum())}")

    # Preliminary analysis to justify using the LRP bounded method for the first layer of the NN
    n = X_full.size
    print(f"Max of input data : {np.max(X_full):2.2f}")
    print(f"Min of input data  : {np.min(X_full):2.2f}")
    print(f"Mean of input data  : {np.mean(X_full):2.2e}")
    print(f"Std of input data  : {np.std(X_full):2.2f}")
    for bound in (1, 2, 3):
        print(f"Percentage of points within [-{bound},{bound}] : {np.count_nonzero(np.abs(X_full) <= bound) / n * 100:2.2f}%")

    if args.no_plots:
        raise SystemExit

    os.makedirs(args.out_dir, exist_ok = True)
    out = lambda name: join(args.out_dir, name)

    # Max, min and standard deviation of each grid point's time series
    stds = np.std(X_full, axis = 0)
    viz.plot_panels([(common.to_llc(np.max(X_full, axis = 0), wetpoints), "Max SST anomaly for each grid cell"   , 'RdBu_r', -5, 5),
                     (common.to_llc(np.min(X_full, axis = 0), wetpoints), "Min SST anomaly for each grid cell"   , 'RdBu_r', -5, 5),
                     (common.to_llc(stds                     , wetpoints), "STD SST anomalies for each grid cell", 'jet'   ,  0, 1)],
                    XC, YC, out('statistics_maps.png'), ncols = 1, figsize = (15,15))

    plt.figure()
    plt.hist(stds, bins = 100)
    plt.title("Histogram of std of each grid-point's time series")
    plt.savefig(out('std_histogram.png'), dpi = 150, bbox_inches = 'tight')
    plt.close('all')

    viz.plot_panels([(common.to_llc(X_full[t], wetpoints), f"SST anomaly {label}", 'RdBu_r', -2, 2)
                     for t, label in ((0, "1992_01_01"), (182, "1992_07_01"), (-365, "2022_01_01"), (-1, "2022_12_31"))],
                    XC, YC, out('anomaly_maps.png'), ncols = 2, figsize = (20,10))

    plt.figure(figsize = (20,10))
    plt.plot(y_full[30:-30], label = "Original new anomalies")
    plt.plot(y, linewidth = 5, label = "61-day convoluted new anomalies")
    plt.title("Effect of convolution on the new anomalies.")
    plt.legend()
    plt.savefig(out('convolution_effect.png'), dpi = 150, bbox_inches = 'tight')
    plt.close('all')

    # Correlation maps, X and y are the ones left after the convolution
    for lags, filename in ((args.lags, 'correlation.png'), (args.long_lags, 'correlation_long_lags.png')):
        corr = viz.correlation_maps(X, y, lags, wetpoints)
        viz.plot_panels(viz.lag_grid(corr, "Correlations lag {lag} days", 'RdBu_r', -0.7, 0.7),
                        XC, YC, out(filename), ncols = 3, figsize = (20,10))
