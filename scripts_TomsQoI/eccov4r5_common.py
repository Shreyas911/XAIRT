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
import sys
import types
from os.path import join
from pathlib import Path

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

def make_layers(n_features, l2_first = 10.0):
    """l2_first: the L2 penalty of the first hidden layer, 10 in the published SST experiment (--l2-first)."""
    return [{'size': n_features, 'activation': None     , 'use_bias': None},
            {'size': 8         , 'activation': 'relu'   , 'use_bias': True,
             'l1_w_reg': 0.0, 'l1_b_reg': 0.0, 'l2_w_reg': l2_first, 'l2_b_reg': l2_first},
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

RESULTS_DIR = Path(__file__).resolve().parent / "results"

def default_out_dir():
    """scripts_TomsQoI/results/<name of the script that runs>, the default --out-dir of every script."""
    return str(RESULTS_DIR / Path(sys.argv[0]).stem)

def parse_args(backend, extra_args = None):
    """extra_args(parser) adds the options of a script, e.g. add_oi_args in eccov4r5_OI_common.py"""
    p = argparse.ArgumentParser(description=f"ECCOv4r5 LRP-A1B0 experiment, {backend} backend.")
    # Sverdrup: /scratch2/pillarh/eccov4r4 and /scratch2/pillarh/eccov4r5 (the GRID and
    # SST_all.nc paths are the notebook's, thetaSurfECCOv4r4.nc was in /scratch2/shreyas/...)
    p.add_argument("--r4-dir", default="/work/07665/shrey911/ls6/LRP_eccov4r4_data",
                   help="directory with thetaSurfECCOv4r4.nc")
    p.add_argument("--r5-dir", default="/work/07665/shrey911/ls6/LRP_eccov4r5_data",
                   help="directory with GRID/ and SST_all.nc")
    p.add_argument("--out-dir", default=default_out_dir(),
                   help="models, results and figures are written here, by default scripts_TomsQoI/results/<script name>")
    p.add_argument("--lags", type=int, nargs="+", default=[-60,-30,0,30,60,90,120,150,180],
                   help="lags in days")
    p.add_argument("--epochs", type=int, default=500, help="use a small number to test the script")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-plots", action="store_true", help="do not save figures")
    p.add_argument("--dataset", choices=sorted(DEFAULT_FIELDS), default="sst",
                   help="'sst' (default): SST anomalies of SST_all.nc, the QoI is the SST anomaly at the objective point (the "
                        "published experiment); 'forcing': the precomputed daily anomalies of --forcing-dir (e.g. wind stress), "
                        "the QoI is the AMOC at RAPID. One dataset at a time, see load_anomalies")
    p.add_argument("--fields", nargs="+", default=None,
                   help="input fields of the dataset, side by side in X: for 'sst' the names of FIELDS (default SST), for "
                        "'forcing' the <name> of its files (default oceTAUE oceTAUN)")
    p.add_argument("--field-scale", choices=["none", "std"], default="none",
                   help="divide each field's anomalies by their std over all grid points and days ('std'), so that fields "
                        "in different units are comparable; 'none' (default) keeps the published SST experiment")
    p.add_argument("--l2-first", type=float, default=10.0,
                   help="L2 penalty of the first hidden layer's weights and biases; 10 (default) is the published SST experiment. "
                        "The forcing fields did not learn with 10 (2026-11-18), see HANDOFF")
    p.add_argument("--forcing-dir", default=FORCING_DIR,
                   help="directory with INPUT_FORCINGS/ and TARGET_QoIs/ of the 'forcing' dataset")
    p.add_argument("--forcing-anomalies", choices=["as-is", "reanomalize"], default="as-is",
                   help="'forcing' dataset: use the anomalies of the files as they are (default), or also remove the mean "
                        "and linear trend of each calendar day with anomalize_new, as for SST")
    if extra_args is not None:
        extra_args(p)
    args = p.parse_args()
    if args.fields is None:
        args.fields = DEFAULT_FIELDS[args.dataset]
    return args

def data_kwargs(args):
    """The keyword arguments of load_data / load_anomalies that choose the dataset."""
    return dict(dataset = args.dataset, forcing_dir = args.forcing_dir, forcing_anomalies = args.forcing_anomalies)

def add_model_args(parser, sources):
    """The options of the scripts that need a trained network for each lag (OI and analyze)."""
    parser.add_argument("--source", choices = sources, default = sources[0],
                        help="where the networks come from: the saved Keras models of --saved-models-dir "
                             "('saved' for Keras, 'saved-keras' to copy them into PyTorch), the saved PyTorch models ('saved-torch', copied into "
                             "Keras with torch_to_keras by the Keras scripts, "
                             "model{lag}_noL1.pt) or 'train' to train them as in the LRP script")
    parser.add_argument("--saved-models-dir",
                        default = "LRP_output_forHelen/saved_models",
                        help="directory with the Keras models model{lag}_noL1.h5, the path of the notebooks")

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

def load_data(r4_dir, r5_dir, fields = ('SST',), field_scale = 'none', **kwargs):
    """X, oneHotCost, wetpoints, XC, YC of load_data_with_y"""
    X, y, oneHotCost, wetpoints, XC, YC = load_data_with_y(r4_dir, r5_dir, fields, field_scale, **kwargs)
    return X, oneHotCost, wetpoints, XC, YC

# --- Input fields -------------------------------------------------------------------------
#
# The input X of the networks is the anomalies of one or more daily surface fields at the wet points, side by side:
# X[:, k*n_wet:(k+1)*n_wet] is field k of --fields (n_wet = len(wetpoints[0]); all fields share the SST wet-point mask, so
# they must be surface fields on the LLC90 grid; a field with its own mask needs a mask per field). To add a field, write a
# loader that returns its daily values (days, 13, 90, 90) on the same days as SST_all.nc, and add it to FIELDS; every script
# then takes it with --fields SST <NAME>. The maps (relevance, OI, composites) are split back per field by field_maps(), and
# with one field the names of the NetCDF variables and figures are the ones of the single-SST experiment.

def load_sst(r4_dir, r5_dir):
    return xr.open_dataset(join(r5_dir, 'SST_all.nc'))['SST'].data

FIELDS = {'SST': load_sst}

# --- The 'forcing' dataset ----------------------------------------------------------------
#
# Daily anomalies precomputed elsewhere (7-day low-pass, about a 366-day climatology), 1992-01-01 to 2021-12-31, one file per
# field, and the AMOC at RAPID as the QoI. Laid out as in sample_LRP_grid_improved_v2.py (D. Apte), the example of these files:
#   INPUT_FORCINGS/<FORCING_FILE of the field>   big-endian float32, (FORCING_N_DAYS, n_wet), the wet points only
#   INPUT_FORCINGS/eccov4r5_iwetC2d.data          big-endian float32, (n_wet,), the position of each column in the flat compact
#                                                 (1170, 90) LLC90 array, counted from 0 or 1 (forcing_index_base)
#   TARGET_QoIs/<FORCING_QOI_FILE>                big-endian float32, (FORCING_N_DAYS,)
# The wet points are the dataset's own (all latitudes), not the SST ones (YC > -20).

DEFAULT_FIELDS = {'sst': ['SST'], 'forcing': ['oceTAUE', 'oceTAUN']}
FORCING_DIR = "/work/07665/shrey911/ls6/LRP_eccov4r5_forcing_data"
FORCING_FILE = "eccov4r5_7daylpf_daily_anomaly_about_366dayclimatology_{}_Jan1992_Dec2021.data"
FORCING_QOI_FILE = FORCING_FILE.format("RAPID_QoI")
FORCING_N_DAYS = 10958
FORCING_N_YEARS = 30
def forcing_index_base(position, grid_dir):
    """
    0 or 1, whichever puts every position on a wet cell of hFacC (surface level, compact layout). The example script used 0
    and found ~2.6% of the cells misplaced, the pattern of a 1-based index read as 0-based; see results/checks/check_forcing_data.py
    (multi_field worktree). Raises if neither base does.
    """
    n_cells = int(np.prod(LLC_SHAPE))
    wet = np.fromfile(join(grid_dir, 'hFacC.data'), dtype = '>f4')[:n_cells] > 0
    for base in (0, 1):
        p = position - base
        if p.min() >= 0 and p.max() < n_cells and wet[p].all():
            print(f"forcing files: iwetC2d positions are {base}-based (all {len(p)} on wet cells)")
            return base
    raise ValueError("iwetC2d positions fall on dry hFacC cells for both base 0 and base 1")

def forcing_wetpoints(forcing_dir, grid_dir):
    """
    The wet points (tile, j, i index arrays, the order of np.nonzero as for SST) of the forcing files, and for each of them the
    column of the files.
    """
    position = np.fromfile(join(forcing_dir, 'INPUT_FORCINGS', 'eccov4r5_iwetC2d.data'), dtype = '>f4').astype(np.int64)
    position -= forcing_index_base(position, grid_dir)
    column = np.full(np.prod(LLC_SHAPE), -1.0)
    column[position] = np.arange(len(position))
    column = np.rint(ecco.llc_compact_to_tiles(column.reshape(LLC_SHAPE[0]*LLC_SHAPE[1], LLC_SHAPE[2]),
                                               less_output = True)).astype(np.int64)
    wetpoints = np.nonzero(column >= 0)

    hFacC = ecco.read_llc_to_tiles(grid_dir, 'hFacC.data', less_output = True)
    n_dry = int((hFacC[wetpoints] <= 0).sum())
    if n_dry > 0:
        raise ValueError(f"{n_dry} of {len(position)} wet points of the forcing files are dry in hFacC after the compact -> tiles "
                         f"conversion")
    return wetpoints, column[wetpoints]

def load_forcing(forcing_dir, grid_dir, fields, forcing_anomalies):
    """X parts (one (days, n_wet) array per field), the QoI (days,) and the wetpoints of the 'forcing' dataset."""
    wetpoints, columns = forcing_wetpoints(forcing_dir, grid_dir)
    parts = []
    for name in fields:
        flat = np.fromfile(join(forcing_dir, 'INPUT_FORCINGS', FORCING_FILE.format(name)), dtype = '>f4')
        part = flat.reshape(FORCING_N_DAYS, -1)[:, columns].astype(np.float32)
        n_bad = int((~np.isfinite(part)).sum())
        if n_bad:
            # As the example script (np.nan_to_num); a network cannot take NaN
            print(f"forcing field {name}: {n_bad} non-finite values ({n_bad/part.size:.2e} of all) set to 0")
            part = np.nan_to_num(part, nan = 0.0, posinf = 0.0, neginf = 0.0)
        parts.append(part)
    y_full = np.fromfile(join(forcing_dir, 'TARGET_QoIs', FORCING_QOI_FILE), dtype = '>f4').astype(np.float32)
    if len(y_full) != FORCING_N_DAYS:
        raise ValueError(f"the QoI has {len(y_full)} days, expected {FORCING_N_DAYS}")
    if forcing_anomalies == 'reanomalize':
        parts = [anomalize_new(part, num_years = FORCING_N_YEARS) for part in parts]
        y_full = anomalize_new(y_full, num_years = FORCING_N_YEARS)
    return parts, y_full, wetpoints

def field_maps(values, wetpoints, fields):
    """
    (len(fields)*n_wet,) -> [(tag, (13, 90, 90) map)], one per field, tag '' for a single field (the names of the single-SST
    experiment) and '<FIELD>_' otherwise, to put into a variable name, e.g. f"lrp_{lag}_{tag}{method}_{cls}".
    """
    n = len(wetpoints[0])
    if len(values) != n*len(fields):
        raise ValueError(f"{len(values)} values for {len(fields)} fields of {n} wet points")
    if len(fields) == 1:
        return [('', to_llc(values, wetpoints))]
    return [(f'{field}_', to_llc(values[k*n:(k+1)*n], wetpoints)) for k, field in enumerate(fields)]

def load_anomalies(r4_dir, r5_dir, fields = ('SST',), field_scale = 'none', dataset = 'sst', forcing_dir = FORCING_DIR,
                   forcing_anomalies = 'as-is'):
    """
    Returns X_full (days, len(fields)*wetpoints) of the new anomalies of the input fields (see FIELDS), y_full (days,) of the
    QoI, the SST anomaly at the grid point of the objective function, the indices of the wetpoints and XC, YC for plotting.
    With field_scale 'std' each field's anomalies are divided by their std over all wet points and days.
    With dataset 'forcing' the fields, the QoI (the AMOC at RAPID) and the wet points are those of load_forcing.

    The notebook also read the mds SST files with xmitgcm and built a number of DataArrays
    (masks, X, y, ...) that the results never used, only XC and YC are still needed.
    """

    gridDir = join(r5_dir, 'GRID')
    ds_r4 = xr.open_dataset(join(r4_dir, 'thetaSurfECCOv4r4.nc'))

    def llc_dataarray(data):
        return xr.DataArray(data = data,
                            dims = ["tile", "j", "i"],
                            coords = dict(tile = ds_r4['tile'].data,
                                          j    = ds_r4['j'].data,
                                          i    = ds_r4['i'].data))
    XC = llc_dataarray(ds_r4['XC'].data)
    YC = llc_dataarray(ds_r4['YC'].data)

    if dataset == 'forcing':
        parts, y_full, wetpoints = load_forcing(forcing_dir, gridDir, fields, forcing_anomalies)
        if field_scale == 'std':
            parts = [part / np.std(part) for part in parts]
        X_full = parts[0] if len(parts) == 1 else np.concatenate(parts, axis = 1)
        return X_full, y_full, wetpoints, XC, YC

    SST = load_sst(r4_dir, r5_dir)

    hFacC = ecco.read_llc_to_tiles(gridDir, 'hFacC.data')
    hFacC_mask = (hFacC > 0).astype(float)

    latMask = (ds_r4['YC'].data > -20.0).astype(float)
    maskFinal = hFacC_mask * latMask
    wetpoints = np.nonzero(maskFinal)

    parts = []
    for name in fields:
        data = SST if name == 'SST' else FIELDS[name](r4_dir, r5_dir)
        if data.shape[0] != SST.shape[0]:
            raise ValueError(f"field {name} has {data.shape[0]} days, SST {SST.shape[0]}")
        part = anomalize_new(data[:,wetpoints[0],wetpoints[1],wetpoints[2]].copy())
        if field_scale == 'std':
            part = part / np.std(part)
        parts.append(part)
    X_full = parts[0] if len(parts) == 1 else np.concatenate(parts, axis = 1)
    y_full = anomalize_new(SST[:,10,1,43].copy())

    return X_full, y_full, wetpoints, XC, YC

def make_qoi(X_full, y_full):
    """
    X (samples, wetpoints), the QoI y (samples,), the running mean of y_full, and the one-hot QoI (samples, 2) as float32.
    The first and last 30 days are left out, since the QoI is a 61-day running mean.
    """

    X = X_full[30:-30]

    # https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
    y = np.convolve(y_full, np.ones(61)/61, mode='valid')

    # float, since the trainers only take float arrays
    oneHotCost = np.zeros((y.shape[0], 2), dtype = np.float32)
    oneHotCost[:,0] = y >= 0.0
    oneHotCost[:,1] = y <  0.0

    return X, y, oneHotCost

def load_data_with_y(r4_dir, r5_dir, fields = ("SST",), field_scale = "none", **kwargs):
    """
    Returns X (samples, len(fields)*wetpoints) of the anomalies, the QoI y (samples,) before it is made a class, the one-hot QoI
    (samples, 2) as float32, the indices of the wetpoints and XC, YC for plotting, see make_qoi. kwargs choose the dataset, see
    load_anomalies.
    """

    X_full, y_full, wetpoints, XC, YC = load_anomalies(r4_dir, r5_dir, fields, field_scale, **kwargs)
    X, y, oneHotCost = make_qoi(X_full, y_full)

    return X, y, oneHotCost, wetpoints, XC, YC

def make_context(args):
    """
    The data and what is needed to train a network of a lag: X, oneHotCost, wetpoints, XC, YC, layers, class_weight,
    idx (the samples before it are for training and validation, the rest is the test set) and models_dir.
    """

    os.makedirs(args.out_dir, exist_ok = True)
    models_dir = join(args.out_dir, 'models')
    os.makedirs(models_dir, exist_ok = True)

    X, oneHotCost, wetpoints, XC, YC = load_data(args.r4_dir, args.r5_dir, args.fields, args.field_scale, **data_kwargs(args))
    idx = int(X.shape[0]*(1-TEST_SPLIT_FRAC))
    # The SST test set is the notebook's last N_TEST samples; another dataset keeps the same test fraction
    n_test = N_TEST if args.dataset == 'sst' else X.shape[0] - idx
    return types.SimpleNamespace(X = X, oneHotCost = oneHotCost, wetpoints = wetpoints, XC = XC, YC = YC, fields = list(args.fields),
                                 layers = make_layers(X.shape[1], args.l2_first),
                                 class_weight = class_weights(oneHotCost, n_test),
                                 idx = idx,
                                 models_dir = models_dir)

def train_on_lag(ctx, lag, train_fn):
    """Trains a network of a lag as the LRP scripts do, train_fn is the make_train_fn(args) of the backend."""
    x_t, x_v, y_t, y_v = split_lag(ctx.X[:ctx.idx], ctx.oneHotCost[:ctx.idx], lag, VAL_SPLIT_FRAC)
    model, _ = train_fn(x_t, y_t, x_v, y_v, lag, ctx.layers, ctx.class_weight, ctx.models_dir)
    return model

def class_weights(oneHotCost, n_test = N_TEST):
    train = oneHotCost[:-n_test]
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
    """
    Each sample divided by its max abs, then the mean over the samples. The notebook (LRP_normalize) divided by the max, which
    breaks the whole map when one sample has relevance <= 0 (max 0 -> inf / NaN everywhere): LRP-A1B0 with the bias kept gives
    that for every sample whose explained logit is negative (job 3545911). For a sample whose max is also its max abs (nearly
    all samples of the published maps, whose relevance is almost all positive) the two are the same. Changed 2026-11-10 by the
    owner's choice. An all-zero sample is still 0/0 = NaN and left out by nanmean.
    """
    return np.nanmean(a / np.nanmax(np.abs(a), axis = 1)[:,np.newaxis], axis = 0)

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

    ctx = make_context(args)
    X, oneHotCost, wetpoints, XC, YC = ctx.X, ctx.oneHotCost, ctx.wetpoints, ctx.XC, ctx.YC
    layers, class_weight, idx, models_dir = ctx.layers, ctx.class_weight, ctx.idx, ctx.models_dir

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

        nan_values = np.full(X.shape[1], np.nan)
        for cls, idx_c in (('pos', idx_pos), ('neg', idx_neg)):
            comp = np.nanmean(X[idx_c], axis = 0) if len(idx_c) > 0 else nan_values
            for tag, field in field_maps(comp, wetpoints, args.fields):
                maps[f'{tag}comp_{cls}'] = field

            for method in methods:
                print(f"Analyze using {method['title']} for {cls} samples")
                mean = normalized_mean_relevance(explain_fn(model_for_xai, method, X[idx_c])) if len(idx_c) > 0 else nan_values
                for tag, field in field_maps(mean, wetpoints, args.fields):
                    maps[f"{tag}{title_key(method['title'])}_{cls}"] = field

        results[lag] = (result, maps)

    finalize(backend, args, results, methods, XC, YC)

def field_tags(fields):
    """The tags of field_maps(): [''] for a single field, ['<FIELD>_', ...] otherwise."""
    return [''] if len(fields) == 1 else [f'{field}_' for field in fields]

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

    for tag in field_tags(args.fields):
        for cls in ('pos', 'neg'):

            plot_maps({lag: maps[f'{tag}comp_{cls}'] for lag, (_, maps) in results.items()}, XC, YC,
                      f"{tag}Composite observations ({cls}_all) lag {{lag}} days",
                      join(args.out_dir, f'{tag}composite_{cls}_{backend}.png'),
                      cmap = cmocean.cm.balance, cmin = -1, cmax = 1)

            for method in methods:
                key = title_key(method['title'])
                plot_maps({lag: maps[f'{tag}{key}_{cls}'] for lag, (_, maps) in results.items()}, XC, YC,
                          f"{tag}{method['title']}_all lag {{lag}} days",
                          join(args.out_dir, f'{tag}{key}_{cls}_{backend}.png'),
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
