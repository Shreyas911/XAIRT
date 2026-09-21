"""
Shared by eccov4r5_analyze_keras.py and eccov4r5_analyze_torch.py, converted from
notebooks_TomsQoI/eccov4r5_classification-analyze-savedNN.ipynb

For each lag the predictions of the trained network of that lag, of the whole series, of the samples it was trained
on (train) and of the ones it was not (test), against the true class, with the accuracy and the F1 score of each. The
data and the experiment are the ones of eccov4r5_common.py. A backend script only has to say how to get a network and
how to predict with it, see run_analyze().

Written to --out-dir, with the variable names of the notebook, {lag} being minus60, minus30, 0, 30 and so on:
  qoi_pred_{backend}.nc     QoI_true_full_{lag}, and QoI_true_trunc_{lag}, QoI_predict_trunc_{lag} and the same with
                            _train_trunc and _test_trunc, the classes (0 is positive, 1 negative). trunc is the
                            samples that have a partner at the lag, see lag_align()
  accuracy_f1_{backend}.nc  accuracy_{lag}, f1score_{lag}, and accuracy_train_{lag}, f1score_train_{lag}, accuracy_test_{lag},
                            f1score_test_{lag}
"""

from os.path import join

import numpy as np
import xarray as xr
from sklearn.metrics import accuracy_score, f1_score

import eccov4r5_common as common

def evaluate(model, predict_fn, X, oneHotCost, lag):
    """The true and predicted class of the samples with a partner at the lag, and the accuracy and F1 score."""
    x_al, y_al = common.lag_align(X, oneHotCost, lag)
    QoI_predict = np.argmax(predict_fn(model, x_al), axis = 1)
    QoI_true = np.argmax(y_al, axis = 1)
    return QoI_true, QoI_predict, accuracy_score(QoI_true, QoI_predict), f1_score(QoI_true, QoI_predict, average = 'binary')

def analyze_lag(model, predict_fn, ctx, lag):
    """The variables of qoi_pred.nc and of accuracy_f1.nc of a lag, without the lag in their names."""

    qoi = {'QoI_true_full': np.argmax(ctx.oneHotCost, axis = 1)}
    metrics = {}
    message = f"Lag {lag}"

    # The whole series, the samples of the training and validation, the test samples
    for part, label, X, oneHotCost in (('',       '',       ctx.X,          ctx.oneHotCost),
                                       ('_train', ' Train', ctx.X[:ctx.idx], ctx.oneHotCost[:ctx.idx]),
                                       ('_test',  ' Test',  ctx.X[ctx.idx:], ctx.oneHotCost[ctx.idx:])):
        QoI_true, QoI_predict, accuracy, f1 = evaluate(model, predict_fn, X, oneHotCost, lag)
        qoi[f"QoI_true{part}_trunc"] = QoI_true
        qoi[f"QoI_predict{part}_trunc"] = QoI_predict
        metrics[f"accuracy{part}"] = accuracy
        metrics[f"f1score{part}"] = f1
        message += f", Accuracy{label} {accuracy*100:2.2f}, F1 Score{label} {f1:2.2f}"

    print(message)

    return qoi, metrics

def run_analyze(backend, args, get_model, predict_fn):
    """
    The whole analysis, with the backend specific parts given as functions:

      get_model(lag, ctx) -> network, see make_get_model in the LRP script of the backend
      predict_fn(model, X) -> class probabilities (samples, 2)
    """

    ctx = common.make_context(args)

    ds_qoi = {}
    ds_metrics = {}

    for lag in args.lags:

        print(f'Lag: {lag} days, for Theta')

        name = common.lag_name(lag)
        qoi, metrics = analyze_lag(get_model(lag, ctx), predict_fn, ctx, lag)

        for key, values in qoi.items():
            # The dimension of a variable is named after the lag and the part of the samples, e.g. minus60_full, minus60_train_trunc
            dim = f"{name}_full" if key == 'QoI_true_full' else f"{name}_" + key.split('_', 2)[2]
            ds_qoi[f"{key}_{name}"] = xr.DataArray(values, dims = (dim,))
        for key, value in metrics.items():
            ds_metrics[f"{key}_{name}"] = xr.DataArray(value)

    xr.Dataset(ds_qoi).to_netcdf(join(args.out_dir, f'qoi_pred_{backend}.nc'))
    xr.Dataset(ds_metrics).to_netcdf(join(args.out_dir, f'accuracy_f1_{backend}.nc'))
