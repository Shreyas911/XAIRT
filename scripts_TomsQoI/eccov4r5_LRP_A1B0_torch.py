"""
ECCOv4r5 classification and LRP with the PyTorch backend (Captum), the counterpart of
eccov4r5_LRP_A1B0_keras.py, which is converted from
notebooks_TomsQoI/eccov4r5_classification-LRP-A1B0-IB-B-newAnomalies-shuffleVal-reweight.ipynb

    python scripts_TomsQoI/eccov4r5_LRP_A1B0_torch.py --out-dir LRP_output_torch
    python scripts_TomsQoI/eccov4r5_LRP_A1B0_torch.py --epochs 2 --lags 0    # quick test

The data, the experiment and the analysis are in eccov4r5_common.py and are shared with the
Keras script, so the two backends run the identical experiment.
"""

import sys
from os.path import join
from pathlib import Path

import numpy as np
import tensorflow as tf
# XAIRT imports TensorFlow too. Keep it off the GPU, which is for PyTorch.
tf.config.set_visible_devices([], 'GPU')
import tensorflow.keras as keras
import torch

# Append the src directory of the repository to sys.path, then import XAIRT
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from XAIRT import TrainTorchFullyConnectedNN, XAITorch, keras_to_torch, metricF1, model_wo_softmax_torch

import eccov4r5_common as common

# Captum has no 'Bounded' input layer rule, so LRP-A1B0-B of the notebook cannot be run here.
# This is the plain method that the Keras script also runs.
METHODS = [dict(name='lrp.alpha_1_beta_0', title = 'LRP-A1B0', optParams = {})]

def make_train_fn(args):

    def train(x_t, y_t, x_v, y_v, lag, layers, class_weight, models_dir):

        T = TrainTorchFullyConnectedNN(x_t, y_t,
                                       layers = layers,
                                       losses = common.LOSSES,
                                       optimizer = 'sgd',    # nesterov, momentum 0.9
                                       learning_rate = 0.01,
                                       metrics = [],         # ignored, accuracy and F1 are computed afterwards
                                       batch_size = common.BATCH_SIZE, epochs = args.epochs,
                                       filename = f'model{lag}_noL1', dirname = models_dir,
                                       validation_data = (x_v, y_v),
                                       random_nn_seed = args.seed, class_weight = class_weight,
                                       verbose = 1)

        best_model = T.quickTrain(common.step_decay)

        # Captum LRP does not support the final Softmax, the layers are shared with best_model
        return best_model, model_wo_softmax_torch(best_model)

    return train

def make_get_model(args):
    """
    get_model(lag, ctx) of the OI and analyze scripts: with --source saved-keras the saved Keras model of the lag, its weights
    copied into a PyTorch model (keras_to_torch), or, with --source train, a model trained with PyTorch.
    """

    train = make_train_fn(args)

    def get_model(lag, ctx):
        if args.source == 'saved-keras':
            keras_model = keras.models.load_model(join(args.saved_models_dir, f'model{lag}_noL1.h5'),
                                                  custom_objects = {'metricF1': metricF1})
            return keras_to_torch(keras_model)
        return common.train_on_lag(ctx, lag, train)

    return get_model

def predict(model, X):
    with torch.no_grad():
        return model(torch.from_numpy(X).float()).numpy()

def explain(model_wo_softmax, method, samples):
    Xplain = XAITorch(model_wo_softmax, method, 'classic', samples, common.NORMALIZE)
    a, _ = Xplain.quick_analyze()
    return a

if __name__ == "__main__":

    args = common.parse_args("torch")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Running on GPU" if torch.cuda.is_available() else "Running on CPU")

    common.run_experiment("torch", args, make_train_fn(args), predict, explain, METHODS)
