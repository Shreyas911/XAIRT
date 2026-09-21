"""
Shared by eccov4r5_results_viz.py and eccov4r5_results_viz_new.py, converted from
notebooks_TomsQoI/eccov4r5_results_viz.ipynb and notebooks_TomsQoI/eccov4r5_results_viz_new.ipynb

Only plotting: nothing here needs Keras or Torch (and XAIRT is not imported, since that would load both),
so these scripts have a single version, not one per backend.
"""

from os.path import exists, join

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import xarray as xr
import ecco_v4_py as ecco

import eccov4r5_common as common

def add_viz_args(parser):
    parser.set_defaults(out_dir = "Results")
    parser.add_argument("--results-dir", default = "LRP_output_forHelen",
                        help="directory with the NetCDF files of the LRP and OI results, the path of the notebooks")

def load_data(args):
    """X, y (samples,), wetpoints, XC, YC, of the new anomalies, the ones of the LRP and OI results."""
    X, y, _, wetpoints, XC, YC = common.load_data_with_y(args.r4_dir, args.r5_dir)
    return X, y, wetpoints, XC, YC

def open_results(results_dir, filename):
    """The dataset of a results file, or None, with a message, if it is not there."""
    path = join(results_dir, filename)
    if not exists(path):
        print(f"{path} not found, its plots are left out")
        return None
    return xr.open_dataset(path)

def field(ds, name):
    """A (13, 90, 90) array of a results dataset, None if the dataset or the variable is not there."""
    return ds[name].values if ds is not None and name in ds else None

def correlation(X, y):
    """Pearson correlation of each column of X with y, as XAIRT.correlation but vectorized."""
    X = X.astype(np.float64)
    Xc = X - X.mean(axis = 0)
    yc = y.astype(np.float64) - y.mean()
    return (Xc*yc[:,np.newaxis]).sum(axis = 0) / np.sqrt((Xc**2).sum(axis = 0) * (yc**2).sum())

def correlation_maps(X, y, lags, wetpoints):
    """{lag: (13, 90, 90)} of the correlation of the SST anomalies with the QoI y, lagged, NaN on land."""
    maps = {}
    for lag in lags:
        x_l, y_l = common.lag_align(X, y, lag)
        maps[lag] = common.to_llc(correlation(x_l, y_l), wetpoints)
    return maps

def plot_panels(panels, XC, YC, path, ncols, figsize):
    """
    One figure with a map for each of the panels, (field, title, cmap, cmin, cmax) each. A field that is None,
    because its file is missing, is left out.
    """

    panels = [p for p in panels if p[0] is not None]
    if not panels:
        return

    nrows = int(np.ceil(len(panels)/ncols))
    plt.figure(figsize = figsize)

    for idx, (fld, title, cmap, cmin, cmax) in enumerate(panels, start = 1):
        P = ecco.plot_proj_to_latlon_grid(XC, YC, fld,
                                          plot_type = 'contourf',
                                          show_colorbar = True,
                                          cmap = cmap,
                                          cmin = cmin,
                                          cmax = cmax,
                                          user_lon_0 = -150,
                                          dx=2, dy=2, projection_type = 'robin',
                                          less_output = True, subplot_grid = [nrows,ncols,idx])
        P[1].set_title(title)

    plt.savefig(path, dpi = 150, bbox_inches = 'tight')
    plt.close('all')

def lag_grid(fields, title, cmap, cmin, cmax):
    """Panels of a field for each lag, {lag: field}, title has a {lag}."""
    return [(fld, title.format(lag = lag), cmap, cmin, cmax) for lag, fld in fields.items()]
