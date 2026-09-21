"""
ECCOv4r5 classification and LRP with the Keras backend (TensorFlow, innvestigate), converted from
notebooks_TomsQoI/eccov4r5_classification-LRP-A1B0-IB-B-newAnomalies-shuffleVal-reweight.ipynb

    python scripts_TomsQoI/eccov4r5_LRP_A1B0_keras.py --out-dir LRP_output_keras
    python scripts_TomsQoI/eccov4r5_LRP_A1B0_keras.py --epochs 2 --lags 0    # quick test

The data, the experiment and the analysis are in eccov4r5_common.py and are shared with
eccov4r5_LRP_A1B0_torch.py, which does the same with PyTorch and Captum.
"""

import sys
from os.path import join
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import innvestigate

# Append the src directory of the repository to sys.path, then import XAIRT
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from XAIRT import TrainKerasFullyConnectedNN, XAIKeras, metricF1

import eccov4r5_common as common

# LRP-A1B0-B is the method of the notebook. Its 'Bounded' input layer rule only exists in
# innvestigate, so LRP-A1B0, plain alpha=1, beta=0 in every layer, is also run: it is the one
# that can be compared with Captum in the Torch script.
METHODS = [dict(name='lrp.alpha_1_beta_0_IB', title = 'LRP-A1B0-B', optParams = {'input_layer_rule':'Bounded'}),
           dict(name='lrp.alpha_1_beta_0'   , title = 'LRP-A1B0'  , optParams = {})]

def make_train_fn(args):

    def train(x_t, y_t, x_v, y_v, lag, layers, class_weight, models_dir):

        keras.backend.clear_session()
        sgd = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

        K = TrainKerasFullyConnectedNN(x_t, y_t,
                                       layers = layers,
                                       losses = common.LOSSES,
                                       optimizer = sgd,
                                       # Custom f1 since metrics.F1Score(name='f1') is not available before tf v2.13
                                       metrics = [metricF1, 'accuracy', 'categorical_crossentropy'],
                                       batch_size = common.BATCH_SIZE, epochs = args.epochs,
                                       filename = f'model{lag}_noL1', dirname = models_dir,
                                       validation_data = (x_v, y_v),
                                       random_nn_seed = args.seed, class_weight = class_weight,
                                       custom_objects = {'metricF1': metricF1}, verbose = 2)

        best_model = K.quickTrain(common.step_decay)

        return best_model, innvestigate.model_wo_softmax(best_model)

    return train

def make_get_model(args):
    """get_model(lag, ctx) of the OI and analyze scripts: the saved Keras model of the lag, or, with --source train, a trained one."""

    train = make_train_fn(args)

    def get_model(lag, ctx):
        if args.source == 'saved':
            return keras.models.load_model(join(args.saved_models_dir, f'model{lag}_noL1.h5'),
                                           custom_objects = {'metricF1': metricF1})
        return common.train_on_lag(ctx, lag, train)

    return get_model

def predict(model, X):
    return model.predict(X)

def explain(model_wo_softmax, method, samples):
    Xplain = XAIKeras(model_wo_softmax, method, 'classic', samples, common.NORMALIZE)
    a, _ = Xplain.quick_analyze()
    return a

if __name__ == "__main__":

    args = common.parse_args("keras")

    ### https://stackoverflow.com/questions/36288235/how-to-get-stable-results-with-tensorflow-setting-random-seed ###
    ### https://keras.io/examples/keras_recipes/reproducibility_recipes/ ###
    keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()

    # Required by innvestigate
    tf.compat.v1.disable_eager_execution()

    print("Running on GPU" if tf.config.list_physical_devices('GPU') else "Running on CPU")

    common.run_experiment("keras", args, make_train_fn(args), predict, explain, METHODS)
