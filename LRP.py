import numpy as np
import scipy.signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import os
from os.path import join, abspath
import sys
import xarray as xr
import ecco_v4_py as ecco
import warnings

# --- 1. Import Torch-Native XAIRT ---
sys.path.insert(0, '/scratch/10027/dhruvapte26/eccoXAI/XAIRT')
from XAIRT.model.Trainer import TrainFullyConnectedNN
from XAIRT.model.XAI import XAIR

warnings.filterwarnings('ignore')

# --- 2. Setup ---
SCRATCH_BASE = '/scratch/10027/dhruvapte26/eccoXAI'
mainDir_r5 = join(SCRATCH_BASE, 'LRP_eccov4r5_data')
gridDir = join(mainDir_r5, 'GRID')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 3. Grid & Mask ---
hFacC = ecco.read_llc_to_tiles(gridDir, 'hFacC.data')
ds_grid_r4 = xr.open_dataset(join(SCRATCH_BASE, 'LRP_eccov4r4_data/thetaSurfECCOv4r4.nc'))
maskFinal = (hFacC[0] > 0).astype(float) * (ds_grid_r4['YC'].data > -20.0).astype(float)
wetpoints = np.nonzero(maskFinal)

# --- 4. Data Logic ---
ds_SST = xr.open_dataset(join(mainDir_r5, 'SST_all.nc'))
SST = ds_SST['SST'].data

def anomalize_new(field, num_years=31):
    leap_yr_offsets_jan_feb   = np.array([0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8])
    leap_yr_offsets_after_feb = np.array([1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8])
    for d in range(59):
        idx = [d + 365*year + leap_yr_offsets_jan_feb[year] for year in range(num_years)]
        field[idx] = scipy.signal.detrend(field[idx], axis=0, type='constant')
        field[idx] = scipy.signal.detrend(field[idx], axis=0, type='linear')
    feb29_idx = [365*year + 59 + int(year/4) for year in range(0, num_years, 4)]
    field[feb29_idx] = scipy.signal.detrend(field[feb29_idx], axis=0, type='constant')
    field[feb29_idx] = scipy.signal.detrend(field[feb29_idx], axis=0, type='linear')
    for d in range(60, 366):
        idx = [d - 1 + 365*year + leap_yr_offsets_after_feb[year] for year in range(num_years)]
        field[idx] = scipy.signal.detrend(field[idx], axis=0, type='constant')
        field[idx] = scipy.signal.detrend(field[idx], axis=0, type='linear')
    return field

X_all = anomalize_new(SST[:, wetpoints[0], wetpoints[1], wetpoints[2]])
X_trimmed = X_all[30:-30]
y_raw = anomalize_new(SST[:, 10, 1, 43].copy())
y_smooth = np.convolve(y_raw, np.ones(61)/61, mode='valid')

oneHotCost = np.zeros((y_smooth.shape[0], 2))
oneHotCost[y_smooth >= 0.0, 0] = 1 # Positive
oneHotCost[y_smooth < 0.0, 1] = 1  # Negative

# --- 5. Main Loop ---
lagStepsList = [-60, -30, 0, 30, 60, 90, 120, 150, 180]
plot_dir = 'LRP_Results_Torch'
os.makedirs(plot_dir, exist_ok=True)

Layers = [{'size': X_trimmed.shape[1], 'activation': None},
          {'size': 8, 'activation': 'relu'},
          {'size': 8, 'activation': 'relu'},
          {'size': 2, 'activation': 'softmax'}]

for lag in lagStepsList:
    print(f"\n--- Running Lag {lag} ---")
    if lag == 0: X_l, y_l = X_trimmed, oneHotCost
    elif lag > 0: X_l, y_l = X_trimmed[:-lag], oneHotCost[lag:]
    else: X_l, y_l = X_trimmed[abs(lag):], oneHotCost[:-abs(lag)]

    trainer = TrainFullyConnectedNN(
        X_l, y_l, layers=Layers, losses=[{'kind': 'CrossEntropyLoss', 'weight': 1.0}],
        optimizer='sgd', metrics=[], batch_size=128, epochs=500,
        filename=f'model_lag{lag}', dirname=abspath('saved_models'), device=device,
        class_weight={0: len(y_l)/np.sum(y_l[:,0]), 1: len(y_l)/np.sum(y_l[:,1])}
    )
    model = trainer.quickTrain()
    model.eval()

    # Predictions for filtering
    with torch.no_grad():
        probs = model(torch.from_numpy(X_l).float().to(device))
        preds = torch.argmax(probs, dim=1).cpu().numpy()
    
    true_labels = np.argmax(y_l, axis=1)

    # Initialize XAIR for LRP
    xair_obj = XAIR(model, method='LRP', kind='LRP', normalize={'bool_': True, 'kind': 'Sum'})

    for class_idx, label_name in zip([0, 1], ['pos', 'neg']):
        # Correct Predictions only
        idx = np.where((true_labels == class_idx) & (preds == class_idx))[0]
        
        if len(idx) > 0:
            print(f"  Generating LRP & Composites for {label_name} ({len(idx)} samples)")
            
            # 1. Composite (Physical Reality)
            comp_avg = np.mean(X_l[idx], axis=0)
            comp_grid = np.full((13, 90, 90), np.nan)
            comp_grid[wetpoints[0], wetpoints[1], wetpoints[2]] = comp_avg
            
            # 2. LRP (Model Interpretation)
            # xair_obj.analyze_samples internally iterates over the batch
            attr_raw = xair_obj.analyze_samples(X_l[idx])
            lrp_avg = np.mean(attr_raw, axis=0)
            lrp_grid = np.full((13, 90, 90), np.nan)
            lrp_grid[wetpoints[0], wetpoints[1], wetpoints[2]] = lrp_avg

            # Plotting Helper
            for data, name, cmap, lim in [(comp_grid, 'Comp', 'RdBu_r', 1.0), 
                                          (lrp_grid, 'LRP', 'RdBu_r', 0.01)]:
                plt.figure(figsize=(15,7))
                ecco.plot_proj_to_latlon_grid(ds_grid_r4.XC, ds_grid_r4.YC, data,
                                              plot_type='contourf', show_colorbar=True,
                                              cmap=cmap, cmin=-lim, cmax=lim,
                                              user_lon_0=-150, dx=2, dy=2,
                                              projection_type='robin', less_output=True)
                plt.title(f"{name} {label_name} Lag {lag}")
                plt.savefig(join(plot_dir, f"{name}_{label_name}_lag{lag}.png"), bbox_inches='tight')
                plt.close()
