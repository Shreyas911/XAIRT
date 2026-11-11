"""
Compares the maps of two NetCDF files written by the ECCOv4r5 scripts (OI, LRP, compare) or the published LRP file: for every
variable that is in both files (after removing --suffix-a / --suffix-b from the names), the Pearson correlation over the grid
points that are finite in both, how many those are, and how many are NaN or inf in each. Prints one line per variable and
writes compare_maps.json to --out-dir (one entry per --pair label).

    python scripts_TomsQoI/eccov4r5_compare_maps.py --out-dir <dir> \
        --pair "keras_vs_torch_trained|<run>/LRP_keras/LRP_..._keras.nc|<run>/LRP_torch/LRP_..._torch.nc" \
        --pair "compare_in_file|<file>.nc|<file>.nc|_keras|_torch"

A --pair is LABEL|FILE_A|FILE_B[|SUFFIX_A|SUFFIX_B]. With the same file twice and two suffixes, it compares the two libraries
of a compare file. Pairs whose files do not exist are reported and skipped, so one job can compare the outputs of several runs.
No ECCO data is read.
"""

import argparse
import json
import os
from os.path import exists, join

import numpy as np
import xarray as xr

def maps(path, suffix):
    ds = xr.open_dataset(path)
    out = {}
    for name in ds.data_vars:
        if suffix and not name.endswith(suffix):
            continue
        out[name[:len(name) - len(suffix)] if suffix else name] = ds[name].values.astype(np.float64).ravel()
    return out

def compare(a, b):
    finite = np.isfinite(a) & np.isfinite(b)
    r = float(np.corrcoef(a[finite], b[finite])[0, 1]) if finite.sum() > 1 and a[finite].std() > 0 and b[finite].std() > 0 \
        else float('nan')
    return {'pearson': r, 'n_finite_both': int(finite.sum()),
            'n_nan_a': int(np.isnan(a).sum()), 'n_inf_a': int(np.isinf(a).sum()),
            'n_nan_b': int(np.isnan(b).sum()), 'n_inf_b': int(np.isinf(b).sum()),
            'max_abs_diff': float(np.abs(a[finite] - b[finite]).max()) if finite.any() else float('nan')}

if __name__ == "__main__":

    p = argparse.ArgumentParser(description = __doc__.split("\n\n")[0])
    p.add_argument("--pair", action = "append", required = True, help="LABEL|FILE_A|FILE_B[|SUFFIX_A|SUFFIX_B]")
    p.add_argument("--out-dir", default = "compare_maps")
    args = p.parse_args()

    results = {}
    for pair in args.pair:
        parts = pair.split("|")
        label, file_a, file_b = parts[:3]
        suffix_a, suffix_b = (parts[3], parts[4]) if len(parts) == 5 else ("", "")
        missing = [f for f in (file_a, file_b) if not exists(f)]
        if missing:
            print(f"{label}: skipped, not found: {missing}", flush = True)
            continue
        a, b = maps(file_a, suffix_a), maps(file_b, suffix_b)
        common_names = [n for n in a if n in b]
        print(f"{label}: {len(common_names)} common maps (A {len(a)}, B {len(b)})", flush = True)
        results[label] = {'file_a': file_a, 'file_b': file_b, 'maps': {}}
        for name in common_names:
            r = compare(a[name], b[name])
            results[label]['maps'][name] = r
            print(f"  {name:32s} pearson {r['pearson']:7.4f}  finite {r['n_finite_both']:6d}  "
                  f"nan/inf A {r['n_nan_a']}/{r['n_inf_a']}  B {r['n_nan_b']}/{r['n_inf_b']}  max|A-B| {r['max_abs_diff']:.3e}",
                  flush = True)

    os.makedirs(args.out_dir, exist_ok = True)
    with open(join(args.out_dir, 'compare_maps.json'), 'w') as f:
        json.dump(results, f, indent = 1)
