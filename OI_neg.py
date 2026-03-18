"""
ecco4r5_classification_OI.py �  PyTorch port of OI_keras_neg.py
"""

import numpy as np
import scipy.signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
from os.path import join, abspath
import sys
import xarray as xr
import ecco_v4_py as ecco
import warnings
from sklearn.model_selection import train_test_split

sys.path.insert(0, '/scratch/10027/dhruvapte26/eccoXAI/XAIRT')
from XAIRT import *

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# 1. Environment & reproducibility
# ---------------------------------------------------------------------------
SEED = 1997
torch.manual_seed(SEED)
np.random.seed(SEED)

SCRATCH_BASE = '/scratch/10027/dhruvapte26/eccoXAI'
mainDir_r5   = join(SCRATCH_BASE, 'LRP_eccov4r5_data')
gridDir      = join(mainDir_r5,   'GRID')
mainDir_r4   = join(SCRATCH_BASE, 'LRP_eccov4r4_data')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# ---------------------------------------------------------------------------
# 2. Grid & data loading
# ---------------------------------------------------------------------------
hFacC      = ecco.read_llc_to_tiles(gridDir, 'hFacC.data')
ds_grid_r4 = xr.open_dataset(join(mainDir_r4, 'thetaSurfECCOv4r4.nc'))
maskFinal  = (hFacC[0] > 0).astype(float) * (ds_grid_r4['YC'].data > -20.0).astype(float)
wetpoints  = np.nonzero(maskFinal)

ds_SST = xr.open_dataset(join(mainDir_r5, 'SST_all.nc'))
SST    = ds_SST['SST'].data
da_XC = xr.DataArray(ds_grid_r4['XC'].data, dims=["tile", "j", "i"], coords=dict(tile=ds_grid_r4['tile'].data, j=ds_grid_r4['j'].data, i=ds_grid_r4['i'].data))
da_YC = xr.DataArray(ds_grid_r4['YC'].data, dims=["tile", "j", "i"], coords=dict(tile=ds_grid_r4['tile'].data, j=ds_grid_r4['j'].data, i=ds_grid_r4['i'].data))

# ---------------------------------------------------------------------------
# 3. anomalize_new
#    Faithful copy from Keras original.
#    hardcoded range(0, num_years, 4), silently shifting which years are leap.
# ---------------------------------------------------------------------------
def anomalize_new(field, num_years=31, first_leap_year_idx=0):
    leap_yr_offsets_jan_feb   = np.array([0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8])
    leap_yr_offsets_after_feb = np.array([1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8])

    for d in range(59):
        idx = [d + 365*yr + leap_yr_offsets_jan_feb[yr] for yr in range(num_years)]
        field[idx] = scipy.signal.detrend(field[idx], axis=0, type='constant', overwrite_data=False)
        field[idx] = scipy.signal.detrend(field[idx], axis=0, type='linear',   overwrite_data=False)

    feb29_idx = [365*yr + 59 + int(yr/4) for yr in range(first_leap_year_idx, num_years, 4)]
    field[feb29_idx] = scipy.signal.detrend(field[feb29_idx], axis=0, type='constant', overwrite_data=False)
    field[feb29_idx] = scipy.signal.detrend(field[feb29_idx], axis=0, type='linear',   overwrite_data=False)

    for d in range(60, 366):
        idx = [d - 1 + 365*yr + leap_yr_offsets_after_feb[yr] for yr in range(num_years)]
        field[idx] = scipy.signal.detrend(field[idx], axis=0, type='constant', overwrite_data=False)
        field[idx] = scipy.signal.detrend(field[idx], axis=0, type='linear',   overwrite_data=False)

    return field

# ---------------------------------------------------------------------------
# 4. Build X and y
#    tried to re-anomalise the target SST index from already-anomalised data.
#    The original anomalises each independently from the raw SST array.
# ---------------------------------------------------------------------------
X_all    = SST[:, wetpoints[0], wetpoints[1], wetpoints[2]].copy()
X_all    = anomalize_new(X_all)
X        = X_all[30:-30]          # trim edges from 61-day convolution

y_raw    = SST[:, 10, 1, 43].copy()
y_raw    = anomalize_new(y_raw)
y_smooth = np.convolve(y_raw, np.ones(61) / 61, mode='valid')

oneHotCost = np.zeros((y_smooth.shape[0], 2), dtype=np.float32)
oneHotCost[:, 0] = (y_smooth >= 0.0).astype(np.float32)   # positive phase
oneHotCost[:, 1] = (y_smooth <  0.0).astype(np.float32)   # negative phase

# ---------------------------------------------------------------------------
# 5. Class weights
#    Computed ONCE on the training portion only, reused for every lag.
#    giving different weights for different lags — inconsistent with original.
# ---------------------------------------------------------------------------
TRAIN_END      = len(oneHotCost) - 2161
weight_for_pos = TRAIN_END / np.sum(oneHotCost[:TRAIN_END, 0])
weight_for_neg = TRAIN_END / np.sum(oneHotCost[:TRAIN_END, 1])
class_weight   = {0: weight_for_pos, 1: weight_for_neg}
print(f"Class weights: pos={weight_for_pos:.4f}, neg={weight_for_neg:.4f}")

# ---------------------------------------------------------------------------
# 6. OI loss function
#    Computed from PRE-SOFTMAX logits to avoid vanishing gradients.
#
#    Why logits?  At high softmax confidence (~0.85), the gradient of BCE
#    w.r.t. the probability is ~0.15, then further squashed by the softmax
#    Jacobian — resulting in near-zero gradients and the input barely moving
#    even after 10,000 steps.  Computing cross-entropy from logits via
#    log_softmax gives gradients proportional to (p - target), which are
#    O(1) at any confidence level.  This matches TF1 graph-mode behaviour
#    from the original Keras script (tf.compat.v1.disable_eager_execution).
# ---------------------------------------------------------------------------
def compute_loss_torch(desired_labels_t: torch.Tensor,
                       pred_logits: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(pred_logits, dim=1)
    return -(desired_labels_t * log_probs).sum(dim=1).mean()

# ---------------------------------------------------------------------------
# 7. OI function
# ---------------------------------------------------------------------------
def OI(model: nn.Module,
       desired_labels_numpy: np.ndarray,
       eta: float,
       max_iters: int,
       print_freq: int,
       inp_numpy: np.ndarray = None,
       convergence_threshold: float = 0.99) -> np.ndarray:
    """
    Optimal Input gradient ascent.

    Parameters
    ----------
    model                 : trained nn.Module (softmax final layer)
    desired_labels_numpy  : shape (1,2) one-hot, e.g. [[0,1]] for neg phase
    eta                   : step size (in normalised-gradient space); 0.001
                            gives ~200-500 iters to reach p=0.99 without
                            driving logits to ±infinity.
    max_iters             : maximum gradient steps
    print_freq            : print prediction every N iterations
    inp_numpy             : (1, n_wetpoints) init; zeros if None
    convergence_threshold : stop when p_target >= this value (default 0.99).
                            Prevents logit saturation and extreme input values.

    Returns
    -------
    np.ndarray of shape (13, 90, 90), NaN outside wet mask
    """
    model.eval()

    if inp_numpy is None:
        inp_numpy = np.zeros((1, X.shape[1]), dtype=np.float32)

    inp_numpy = np.atleast_2d(inp_numpy).astype(np.float32)

    if desired_labels_numpy.ndim == 1:
        desired_labels_numpy = desired_labels_numpy[np.newaxis, :]
    desired_labels_numpy = desired_labels_numpy.astype(np.float32)

    desired_labels_t = torch.tensor(desired_labels_numpy, dtype=torch.float32,
                                    device=device)
    target_class = int(np.argmax(desired_labels_numpy))

    # Strip the final Softmax layer so OI loss operates on raw logits.
    # model is a flat nn.Sequential, so children() yields the layer objects
    # directly.  We build a new Sequential sharing the same layer objects
    # (weights are shared, gradients flow through correctly).
    children = list(model.children())
    if isinstance(children[-1], nn.Softmax):
        logit_model = nn.Sequential(*children[:-1]).to(device)
    else:
        logit_model = model

    inp_np = inp_numpy.copy()

    with torch.no_grad():
        p0 = model(torch.tensor(inp_np, dtype=torch.float32, device=device))
    print(f"Desired label : {desired_labels_numpy}")
    print(f"Iter 0, Prediction {p0.cpu().numpy()}")

    for i in range(max_iters):
        inp_t = torch.tensor(inp_np, dtype=torch.float32,
                             device=device, requires_grad=True)
        with torch.enable_grad():
            logits = logit_model(inp_t)
            loss   = compute_loss_torch(desired_labels_t, logits)
            loss.backward()

        with torch.no_grad():
            grad = inp_t.grad.cpu().numpy()
            # Normalise gradient to unit L2 norm before stepping.
            # This makes eta a true "distance per step" in input space,
            # independent of model confidence or gradient magnitude.
            # Without normalisation, step size varies wildly across lags
            # (well-trained lags have large gradients, poorly-trained ones
            # near-zero), leading to inconsistent spatial patterns.
            grad_norm = np.linalg.norm(grad) + 1e-8
            inp_np[0] -= eta * (grad[0] / grad_norm)

        if (i + 1) % print_freq == 0:
            with torch.no_grad():
                p = model(torch.tensor(inp_np, dtype=torch.float32, device=device))
            p_np = p.cpu().numpy()
            print(f"Iter {i+1}, Prediction {p_np}")
            # Early stop: no benefit continuing past convergence threshold —
            # further steps only push logits toward ±∞ and distort the map.
            if p_np[0, target_class] >= convergence_threshold:
                print(f"  Converged at iter {i+1} "
                      f"(p={p_np[0, target_class]:.4f} >= {convergence_threshold})")
                break

    optimal_input = np.full((13, 90, 90), np.nan, dtype=np.float64)
    optimal_input[wetpoints[0], wetpoints[1], wetpoints[2]] = inp_np[0]
    return optimal_input

# ---------------------------------------------------------------------------
# 8. LR schedule  exact replica of Keras step_decay
# ---------------------------------------------------------------------------
def step_decay(epoch: int) -> float:
    initial_lrate = 0.01
    drop          = 0.5
    epochs_drop   = 50
    return initial_lrate * (drop ** np.floor((1 + epoch) / epochs_drop))

# ---------------------------------------------------------------------------
# 9. Network architecture
# ---------------------------------------------------------------------------
Layers = [
    {'size': X.shape[1], 'activation': None,     'use_bias': None},
    {'size': 8,          'activation': 'relu',    'use_bias': True,
     'l1_w_reg': 0.0,   'l1_b_reg': 0.0,  'l2_w_reg': 10.0,  'l2_b_reg': 10.0},
    {'size': 8,          'activation': 'relu',    'use_bias': True,
     'l1_w_reg': 0.0,   'l1_b_reg': 0.0,  'l2_w_reg': 0.01,  'l2_b_reg': 0.01},
    {'size': 2,          'activation': 'softmax', 'use_bias': True,
     'l1_w_reg': 0.0,   'l1_b_reg': 0.0,  'l2_w_reg': 0.01,  'l2_b_reg': 0.01},
]

Losses = [{'kind': 'crossentropy', 'weight': 1.0}]

# ---------------------------------------------------------------------------
# 10. quickSetup
# ---------------------------------------------------------------------------
def quickSetup(X_full: np.ndarray,
               lagSteps: int,
               test_split_frac: float = 2161.0 / 11263.0,
               val_split_frac: float  = 0.2,
               OI_eta: float          = 0.001,
               OI_epochs: int         = 10000,
               OI_print_freq: int     = 200,
               decay_func             = None,
               init_type: str         = 'neg') -> np.ndarray:
    """
    Train the NN for one lag, pick correctly-classified negative samples,
    initialise OI from their mean, run OI, return the (13,90,90) result.
    """

    # --- temporal train/test split (no shuffle) ---
    n_total   = X_full.shape[0]
    n_test    = int(round(n_total * test_split_frac))
    idx_split = n_total - n_test

    # --- apply lag ---
    if lagSteps > 0:
        X_lag = X_full[:-lagSteps]
        y_lag = oneHotCost[lagSteps:]
    elif lagSteps == 0:
        X_lag = X_full
        y_lag = oneHotCost
    else:  # lagSteps < 0
        X_lag = X_full[-lagSteps:]
        y_lag = oneHotCost[:lagSteps]

    idx_split_lag = min(idx_split, len(X_lag))
    X_train_lag   = X_lag[:idx_split_lag]
    y_train_lag   = y_lag[:idx_split_lag]

    # --- train / val split ---
    x_t, x_v, y_t, y_v = train_test_split(
        X_train_lag, y_train_lag,
        test_size=val_split_frac,
        shuffle=True,
        random_state=42
    )

    trainer = TrainFullyConnectedNN(
        x_t, y_t,
        validation_data=(x_v, y_v),
        layers=Layers,
        losses=Losses,
        optimizer='sgd',
        metrics=[metricF1],
        batch_size=128,
        epochs=500,
        filename=f'model{lagSteps}_OI_neg',
        dirname=abspath('saved_models'),
        random_nn_seed=42,
        class_weight=class_weight,
        verbose=1,
    )

    best_model = trainer.quickTrain(decay_func=decay_func)
    best_model.eval()
    best_model.to(device)

    # --- predictions ---
    with torch.no_grad():
        pred_probs  = best_model(
            torch.tensor(X_lag, dtype=torch.float32, device=device)
        ).cpu().numpy()

    pred_labels = (pred_probs > 0.5).astype(int)
    true_labels = y_lag.astype(int)

    # --- correctly classified negative samples ---
    idx_NN_neg = [i for i in range(len(true_labels))
                  if true_labels[i, 1] == 1 and pred_labels[i, 1] == 1]

    print(f"  Lag {lagSteps}: {len(idx_NN_neg)} correctly classified negative samples")

    if len(idx_NN_neg) == 0:
        print(f"  WARNING: no correctly classified negatives for lag={lagSteps}. "
              "Returning NaN grid.")
        return np.full((13, 90, 90), np.nan)

    neg_avg = np.nanmean(X_lag[idx_NN_neg], axis=0, keepdims=True)

    # --- OI initialisation ---
    if init_type == 'neg':
        inp_init = neg_avg
    elif init_type == 'pos':
        idx_NN_pos = [i for i in range(len(true_labels))
                      if true_labels[i, 0] == 1 and pred_labels[i, 0] == 1]
        inp_init = (np.nanmean(X_lag[idx_NN_pos], axis=0, keepdims=True)
                    if idx_NN_pos else None)
    else:
        inp_init = None

    return OI(best_model,
              desired_labels_numpy=np.array([[0.0, 1.0]]),
              eta=OI_eta,
              max_iters=OI_epochs,
              print_freq=OI_print_freq,
              inp_numpy=inp_init,
              convergence_threshold=0.99)

# ---------------------------------------------------------------------------
# 11. Main loop
# ---------------------------------------------------------------------------
OI_eta        = 0.001   # normalised-gradient step; ~200-500 iters to p=0.99
OI_epochs     = 10000
OI_print_freq = 200

lagStepsList  = [-60, -30, 0, 30, 60, 90, 120, 150, 180]
OI_dict       = {}

os.makedirs('OI_plots_Vista', exist_ok=True)
os.makedirs('saved_models',   exist_ok=True)

for lag in lagStepsList:
    print(f'\n--- Lag: {lag} days ---')
    OI_dict[f'lag{lag}'] = quickSetup(
        X,
        lagSteps      = lag,
        OI_eta        = OI_eta,
        OI_epochs     = OI_epochs,
        OI_print_freq = OI_print_freq,
        decay_func    = step_decay,
        init_type     = 'neg',
    )

# ---------------------------------------------------------------------------
# 12. Plotting
# ---------------------------------------------------------------------------
plt.figure(figsize=(20, 10))
subplot_idx = 1

for lag in lagStepsList:
    # 1. RESHAPE: Ensure data is (13, 90, 90)
    # If OI_dict is currently a flat array (105300 elements), reshape it:
    data_to_plot = OI_dict[f'lag{lag}'].reshape((13, 90, 90))
    
    # 2. PLOT: Use the grid coordinates (ds_grid_r4) and the reshaped data
    P = ecco.plot_proj_to_latlon_grid(
        ds_grid_r4.XC, ds_grid_r4.YC,
        data_to_plot, # <--- Use the reshaped data here!
        plot_type='contourf', show_colorbar=True,
        cmap='RdBu_r', cmin=-1, cmax=1,
        user_lon_0=-150, dx=2, dy=2,
        projection_type='robin', less_output=True,
        subplot_grid=[3, 3, subplot_idx],
    )
    P[1].set_title(f"OI_neg lag {lag} days")
    subplot_idx += 1

plt.savefig('OI_plots_Vista/OI_neg_allLags.png', bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------
# 13. Save to NetCDF
# ---------------------------------------------------------------------------
def lag_to_key(l):
    return str(l).replace('-', 'minus')

ds_oi = xr.Dataset({
    f'OI_{lag_to_key(l)}': xr.DataArray(OI_dict[f'lag{l}'], dims=['tile', 'j', 'i'])
    for l in lagStepsList if f'lag{l}' in OI_dict
})
ds_oi.to_netcdf('OI_v4r5_neg_XAIRT_Torch_Final.nc')
print("Saved OI_v4r5_neg_XAIRT_Torch_Final.nc")
