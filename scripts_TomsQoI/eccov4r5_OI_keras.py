"""
ECCOv4r5 optimal input (OI) with the Keras backend (TensorFlow), converted from
notebooks_TomsQoI/eccov4r5_classification-OI-{pos,neg}-newAnomalies-shuffleVal-reweight-avgInit-savedNN.ipynb

    python scripts_TomsQoI/eccov4r5_OI_keras.py --out-dir OI_output_keras
    python scripts_TomsQoI/eccov4r5_OI_keras.py --lags 0 --kind pos --oi-iters 200 --no-plots    # quick test

The networks are the saved Keras models of --saved-models-dir, as in the notebooks, or with --source train they are
trained as in eccov4r5_LRP_A1B0_keras.py. Eager execution, the default, is used, which is what the notebooks
need for the gradient of the loss w.r.t. the input.

The data, the experiment and the OI are in eccov4r5_common.py and eccov4r5_OI_common.py and are shared with
eccov4r5_OI_torch.py, which does the same with PyTorch.
"""

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

# Append the src directory of the repository to sys.path, then import XAIRT
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from XAIRT import Keras_GradientDescent_useGradientTape, tf_to_numpy

import eccov4r5_common as common
import eccov4r5_OI_common as oi_common
import eccov4r5_LRP_A1B0_keras as keras_script

def optimal_input(model, desired_labels, init, eta, n_iters, print_freq):

    inp_numpy = init.astype(np.float32)
    desired = tf.convert_to_tensor(desired_labels)
    bce = keras.losses.BinaryCrossentropy()

    print(f"Desired label : {desired_labels}")
    print(f"Iter 0, Prediction {model.predict(inp_numpy, verbose = 0)}")

    for i in range(n_iters):
        grads = Keras_GradientDescent_useGradientTape(model, tf.convert_to_tensor(inp_numpy), desired, bce)
        inp_numpy[0,:] = inp_numpy[0,:] - eta*np.squeeze(tf_to_numpy(grads))
        if (i+1) % print_freq == 0:
            print(f"Iter {i+1}, Prediction {model.predict(inp_numpy, verbose = 0)}")

    return inp_numpy[0]

if __name__ == "__main__":

    args = common.parse_args("keras", lambda p: oi_common.add_oi_args(p, ('saved', 'train')))

    ### https://stackoverflow.com/questions/36288235/how-to-get-stable-results-with-tensorflow-setting-random-seed ###
    ### https://keras.io/examples/keras_recipes/reproducibility_recipes/ ###
    keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()

    print("Running on GPU" if tf.config.list_physical_devices('GPU') else "Running on CPU")

    oi_common.run_oi("keras", args, keras_script.make_get_model(args), keras_script.predict, optimal_input)
