"""
True against predicted QoI, converted from notebooks_TomsQoI/results_viz_qoi_pred.ipynb

    python scripts_TomsQoI/results_viz_qoi_pred.py --results-dir LRP_output_forHelen --lags -60 -30

Plots where the predicted class of the network differs from the true one (1 where it does) over the samples, for
each lag, and prints the fraction that is wrong. Only plotting, nothing here needs Keras or Torch, so this script has
a single version. It reads qoi_pred.nc (--file) of --results-dir, with QoI_true_trunc_{lag} and QoI_predict_trunc_{lag} in it, where
{lag} is minus60, minus30, 0, 30 and so on. That file is written by notebooks_TomsQoI/eccov4r5_classification-analyze-savedNN.ipynb, which is eccov4r5_analyze_{keras,torch}.py.
Written to --out-dir: qoi_pred_mismatch.png
"""

import argparse
import os
from os.path import join

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import xarray as xr

def lag_name(lag):
    return f"minus{-lag}" if lag < 0 else str(lag)

if __name__ == "__main__":

    p = argparse.ArgumentParser(description="True against predicted QoI.")
    p.add_argument("--results-dir", default = "LRP_output_forHelen", help="directory with the file")
    p.add_argument("--file", default = "qoi_pred.nc", help="the notebook name, eccov4r5_analyze_{keras,torch}.py write qoi_pred_{keras,torch}.nc")
    p.add_argument("--out-dir", default = "Results")
    p.add_argument("--lags", type = int, nargs = "+", default = [-60, -30], help="lags in days, the notebook has -60 and -30")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok = True)
    ds = xr.open_dataset(join(args.results_dir, args.file))

    fig, axes = plt.subplots(len(args.lags), 1, figsize = (10, 3*len(args.lags)), squeeze = False)

    for ax, lag in zip(axes[:,0], args.lags):
        mismatch = ds[f"QoI_true_trunc_{lag_name(lag)}"].data != ds[f"QoI_predict_trunc_{lag_name(lag)}"].data
        print(f"Lag {lag} days: {mismatch.mean():.4f} of the samples are predicted wrongly")
        ax.plot(mismatch)
        ax.set_title(f"Predicted class differs from the true one, lag {lag} days")
        ax.set_xlabel("sample")

    fig.tight_layout()
    fig.savefig(join(args.out_dir, 'qoi_pred_mismatch.png'), dpi = 150)
