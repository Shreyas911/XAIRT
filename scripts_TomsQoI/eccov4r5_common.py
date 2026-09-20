"""
Shared by eccov4r5_LRP_A1B0_keras.py and eccov4r5_LRP_A1B0_torch.py, converted from
notebooks_TomsQoI/eccov4r5_classification-LRP-A1B0-IB-B-newAnomalies-shuffleVal-reweight.ipynb

Everything that has to be identical between the two backends lives here: the data, the
experiment definition (network, loss, learning rate schedule, splits) and the analysis of
the results. A backend script only has to say how to train a network, how to predict with
it and how to explain it, see run_experiment().
"""

import argparse
import json
import os
from os.path import join

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import scipy.signal
import xarray as xr
import cmocean
import ecco_v4_py as ecco
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# --- Experiment definition, the same for both backends ---------------------------------

LLC_SHAPE = (13, 90, 90)                    # tile, j, i
N_TEST = 2161                               # the last samples are the test set
TEST_SPLIT_FRAC = 2161.0/11263.0
VAL_SPLIT_FRAC = 0.2
BATCH_SIZE = 128
LOSSES = [{'kind': 'categorical_crossentropy', 'weight': 1.0}]
NORMALIZE = {'bool_': True, 'kind': 'MaxAbs'}

def make_layers(n_features):
    return [{'size': n_features, 'activation': None     , 'use_bias': None},
            {'size': 8         , 'activation': 'relu'   , 'use_bias': True,
             'l1_w_reg': 0.0, 'l1_b_reg': 0.0, 'l2_w_reg': 10.0, 'l2_b_reg': 10.0},
            {'size': 8         , 'activation': 'relu'   , 'use_bias': True,
             'l1_w_reg': 0.0, 'l1_b_reg': 0.0, 'l2_w_reg': 0.01, 'l2_b_reg': 0.01},
            {'size': 2         , 'activation': 'softmax', 'use_bias': True,
             'l1_w_reg': 0.0, 'l1_b_reg': 0.0, 'l2_w_reg': 0.01, 'l2_b_reg': 0.01, 'bias_constraint': None}]

# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 50
    lrate = initial_lrate * drop**np.floor((1+epoch)/epochs_drop)
    return lrate

def parse_args(backend):
    p = argparse.ArgumentParser(description=f"ECCOv4r5 LRP-A1B0 experiment, {backend} backend.")
    # Sverdrup: /scratch2/pillarh/eccov4r4 and /scratch2/pillarh/eccov4r5 (the GRID and
    # SST_all.nc paths are the notebook's, thetaSurfECCOv4r4.nc was in /scratch2/shreyas/...)
    p.add_argument("--r4-dir", default="/work/07665/shrey911/ls6/LRP_eccov4r4_data",
                   help="directory with thetaSurfECCOv4r4.nc")
    p.add_argument("--r5-dir", default="/work/07665/shrey911/ls6/LRP_eccov4r5_data",
                   help="directory with GRID/ and SST_all.nc")
    p.add_argument("--out-dir", default=f"LRP_output_{backend}",
                   help="models, results and figures are written here")
    p.add_argument("--lags", type=int, nargs="+", default=[-60,-30,0,30,60,90,120,150,180],
                   help="lags in days")
    p.add_argument("--epochs", type=int, default=500, help="use a small number to test the script")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-plots", action="store_true", help="do not save figures")
    return p.parse_args()

# --- Data ---------------------------------------------------------------------------------

def anomalize_new(field, num_years = 31, first_leap_year_idx = 0):

    leap_yr_offsets_jan_feb   = np.array([0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8])
    leap_yr_offsets_after_feb = np.array([1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8])

    if len(field.shape) > 1:
        seasonal_trend = np.zeros((366, field.shape[1]))
    else:
        seasonal_trend = np.zeros((366,))

    #### Calculate seasonal trend

    # Jan 1 - Feb 28
    for d in range(59):
        same_cal_days_idx=[d+365*year+leap_yr_offsets_jan_feb[year] for year in range(num_years)]
        # Remove mean
        field[same_cal_days_idx] = scipy.signal.detrend(field[same_cal_days_idx],
                                                        axis=0,
                                                        type='constant',
                                                        overwrite_data=False)
        # Remove linear trend
        field[same_cal_days_idx] = scipy.signal.detrend(field[same_cal_days_idx],
                                                        axis=0,
                                                        type='linear',
                                                        overwrite_data=False)

    # Feb 29 starting 1996, so year 2 in 0-indexing
    same_cal_days_idx=[365*year+59+int(year/4) for year in range(first_leap_year_idx,num_years,4)]
    # Remove mean
    field[same_cal_days_idx] = scipy.signal.detrend(field[same_cal_days_idx],
                                                    axis=0,
                                                    type='constant',
                                                    overwrite_data=False)
    # Remove linear trend
    field[same_cal_days_idx] = scipy.signal.detrend(field[same_cal_days_idx],
                                                    axis=0,
                                                    type='linear',
                                                    overwrite_data=False)

    # Mar 1 - Dec 31
    for d in range(60,366):
        same_cal_days_idx=[d-1+365*year+leap_yr_offsets_after_feb[year] for year in range(num_years)]
        # Remove mean
        field[same_cal_days_idx] = scipy.signal.detrend(field[same_cal_days_idx],
                                                        axis=0,
                                                        type='constant',
                                                        overwrite_data=False)
        # Remove linear trend
        field[same_cal_days_idx] = scipy.signal.detrend(field[same_cal_days_idx],
                                                        axis=0,
                                                        type='linear',
                                                        overwrite_data=False)

    return field

def load_data(r4_dir, r5_dir):
    """
    Returns X (samples, wetpoints) of SST anomalies, the one-hot QoI (samples, 2) as float32,
    the indices of the wetpoints and XC, YC for plotting.

    The notebook also read the mds SST files with xmitgcm and built a number of DataArrays
    (masks, X, y, ...) that the results never used, only XC and YC are still needed.
    """

    gridDir = join(r5_dir, 'GRID')
    ds_r4 = xr.open_dataset(join(r4_dir, 'thetaSurfECCOv4r4.nc'))
    SST = xr.open_dataset(join(r5_dir, 'SST_all.nc'))['SST'].data

    hFacC = ecco.read_llc_to_tiles(gridDir, 'hFacC.data')
    hFacC_mask = (hFacC > 0).astype(float)

    latMask = (ds_r4['YC'].data > -20.0).astype(float)
    maskFinal = hFacC_mask * latMask
    wetpoints = np.nonzero(maskFinal)

    def llc_dataarray(data):
        return xr.DataArray(data = data,
                            dims = ["tile", "j", "i"],
                            coords = dict(tile = ds_r4['tile'].data,
                                          j    = ds_r4['j'].data,
                                          i    = ds_r4['i'].data))
    XC = llc_dataarray(ds_r4['XC'].data)
    YC = llc_dataarray(ds_r4['YC'].data)

    X = SST[:,wetpoints[0],wetpoints[1],wetpoints[2]].copy()
    X = anomalize_new(X)
    X = X[30:-30]

    y = SST[:,10,1,43].copy()
    y = anomalize_new(y)
    # https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
    y = np.convolve(y, np.ones(61)/61, mode='valid')

    # float, since the trainers only take float arrays
    oneHotCost = np.zeros((y.shape[0], 2), dtype = np.float32)
    oneHotCost[:,0] = y >= 0.0
    oneHotCost[:,1] = y <  0.0

    return X, oneHotCost, wetpoints, XC, YC

def class_weights(oneHotCost):
    train = oneHotCost[:-N_TEST]
    weight_for_pos = len(train[:,0]) / np.sum(train[:,0])
    weight_for_neg = len(train[:,1]) / np.sum(train[:,1])
    return {0: float(weight_for_pos), 1: float(weight_for_neg)}

def lag_align(X, oneHotCost, lagSteps):
    """The rows of X and the rows of the QoI that they are used to predict."""
    if lagSteps > 0:
        return X[:-lagSteps], oneHotCost[lagSteps:]
    elif lagSteps == 0:
        return X, oneHotCost
    else:
        return X[-lagSteps:], oneHotCost[:lagSteps]

def split_lag(X_train, oneHotCost_train, lagSteps, val_split_frac, seed = 42):
    """Returns x_t, x_v, oneHotCost_t, oneHotCost_v"""
    x, y = lag_align(X_train, oneHotCost_train, lagSteps)
    return train_test_split(x, y, test_size=val_split_frac, shuffle=True, random_state=seed)

def correct_indices(probs, y_true, lagSteps):
    """Indices into X of the samples whose class (pos = 0, neg = 1) is predicted correctly."""

    pred = probs > 0.5

    idx_pos = []
    idx_neg = []

    if lagSteps >= 0:
        for i in range(len(y_true[lagSteps:,0])):
            if y_true[lagSteps+i,0] == 1 and pred[i,0]:
                idx_pos.append(i)
            if y_true[lagSteps+i,1] == 1 and pred[i,1]:
                idx_neg.append(i)
    else:
        for i in range(len(y_true[:lagSteps,0])):
            if y_true[i,0] == 1 and pred[i-lagSteps,0]:
                idx_pos.append(i-lagSteps)
            if y_true[i,1] == 1 and pred[i-lagSteps,1]:
                idx_neg.append(i-lagSteps)

    return idx_pos, idx_neg

# --- Analysis, in wetpoint space and only then mapped back onto the grid --------------------

def to_llc(values, wetpoints):
    """(wetpoints,) -> (13, 90, 90), NaN on land."""
    field = np.full(LLC_SHAPE, np.nan)
    field[wetpoints[0], wetpoints[1], wetpoints[2]] = values
    return field

def normalized_mean_relevance(a):
    """Each sample divided by its max, then the mean over the samples (LRP_normalize in the notebook)."""
    return np.nanmean(a / np.nanmax(a, axis = 1)[:,np.newaxis], axis = 0)

def title_key(title):
    """'LRP-A1B0-B' -> 'a1b0_b'"""
    return title.lower().removeprefix('lrp-').replace('-', '_')

def lag_name(lag):
    return f"minus{-lag}" if lag < 0 else str(lag)

def run_experiment(backend, args, train_fn, predict_fn, explain_fn, methods):
    """
    The whole experiment, with the backend specific parts given as functions:

      train_fn(x_t, y_t, x_v, y_v, lag, layers, class_weight, models_dir) -> (model, model_for_xai)
      predict_fn(model, X) -> class probabilities (samples, 2)
      explain_fn(model_for_xai, method, samples) -> relevance (samples, wetpoints)
    """

    os.makedirs(args.out_dir, exist_ok = True)
    models_dir = join(args.out_dir, 'models')
    os.makedirs(models_dir, exist_ok = True)

    X, oneHotCost, wetpoints, XC, YC = load_data(args.r4_dir, args.r5_dir)
    layers = make_layers(X.shape[1])
    class_weight = class_weights(oneHotCost)

    idx = int(X.shape[0]*(1-TEST_SPLIT_FRAC))
    nan_map = np.full(LLC_SHAPE, np.nan)

    results = {}
    for lag in args.lags:

        print(f'Lag: {lag} days, for Theta')

        x_t, x_v, y_t, y_v = split_lag(X[:idx], oneHotCost[:idx], lag, VAL_SPLIT_FRAC)
        model, model_for_xai = train_fn(x_t, y_t, x_v, y_v, lag, layers, class_weight, models_dir)

        X_al, y_al = lag_align(X, oneHotCost, lag)
        QoI_predict = np.argmax(predict_fn(model, X_al), axis=1)
        QoI_true = np.argmax(y_al, axis=1)

        idx_pos, idx_neg = correct_indices(predict_fn(model, X), oneHotCost, lag)

        result = {'Accuracy': float(accuracy_score(QoI_true, QoI_predict)),
                  'F1_Score': float(f1_score(QoI_true, QoI_predict, average='binary')),
                  'n_pos': len(idx_pos), 'n_neg': len(idx_neg)}
        maps = {}

        for cls, idx_c in (('pos', idx_pos), ('neg', idx_neg)):
            maps[f'comp_{cls}'] = to_llc(np.nanmean(X[idx_c], axis = 0), wetpoints) if len(idx_c) > 0 else nan_map

            for method in methods:
                print(f"Analyze using {method['title']} for {cls} samples")
                if len(idx_c) > 0:
                    a = explain_fn(model_for_xai, method, X[idx_c])
                    maps[f"{title_key(method['title'])}_{cls}"] = to_llc(normalized_mean_relevance(a), wetpoints)
                else:
                    maps[f"{title_key(method['title'])}_{cls}"] = nan_map

        results[lag] = (result, maps)

    finalize(backend, args, results, methods, XC, YC)

def finalize(backend, args, results, methods, XC, YC):

    metrics = {str(lag): result for lag, (result, _) in results.items()}
    with open(join(args.out_dir, f'metrics_{backend}.json'), 'w') as f:
        json.dump(metrics, f, indent = 2)

    for lag, result in metrics.items():
        print(f"LRP{lag}")
        print(f"Accuracy: {result['Accuracy']}")
        print(f"F1_Score: {result['F1_Score']}")

    # The same variable names as in the notebook, e.g. lrp_minus60_a1b0_b_pos_all, lrp_0_comp_neg_all
    ds_lrp = xr.Dataset({f"lrp_{lag_name(lag)}_{key}_all": xr.DataArray(field)
                         for lag, (_, maps) in results.items() for key, field in maps.items()})
    ds_lrp.to_netcdf(join(args.out_dir, f'LRP_A1B0_newAnomalies_shuffleVal_reweight_{backend}.nc'))

    if args.no_plots:
        return

    for cls in ('pos', 'neg'):

        plot_maps({lag: maps[f'comp_{cls}'] for lag, (_, maps) in results.items()}, XC, YC,
                  f"Composite observations ({cls}_all) lag {{lag}} days",
                  join(args.out_dir, f'composite_{cls}_{backend}.png'),
                  cmap = cmocean.cm.balance, cmin = -1, cmax = 1)

        for method in methods:
            key = title_key(method['title'])
            plot_maps({lag: maps[f'{key}_{cls}'] for lag, (_, maps) in results.items()}, XC, YC,
                      f"{method['title']}_all lag {{lag}} days",
                      join(args.out_dir, f'{key}_{cls}_{backend}.png'),
                      cmap = 'jet', cmin = 0.0, cmax = 0.3)

def plot_maps(fields, XC, YC, title, path, cmap, cmin, cmax):
    """One figure with a map for each lag, in 3 columns (3x3 for the 9 lags of the notebook)."""

    plt.rcParams["figure.figsize"] = (20,10)
    plt.figure()
    nrows = int(np.ceil(len(fields)/3))

    for subplot_idx, (lag, field) in enumerate(fields.items(), start = 1):
        P = ecco.plot_proj_to_latlon_grid(XC, YC,
                                          field,
                                          plot_type = 'contourf',
                                          show_colorbar = True,
                                          cmap = cmap,
                                          cmin = cmin,
                                          cmax = cmax,
                                          user_lon_0 = -150,
                                          dx=2, dy=2, projection_type = 'robin',
                                          less_output = True, subplot_grid = [nrows,3,subplot_idx])
        P[1].set_title(title.format(lag = lag))

    plt.savefig(path, dpi = 150, bbox_inches = 'tight')
    plt.close('all')
