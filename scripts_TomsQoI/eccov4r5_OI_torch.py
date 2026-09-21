"""
ECCOv4r5 optimal input (OI) with the PyTorch backend, the counterpart of eccov4r5_OI_keras.py, which is converted from
notebooks_TomsQoI/eccov4r5_classification-OI-{pos,neg}-newAnomalies-shuffleVal-reweight-avgInit-savedNN.ipynb

    python scripts_TomsQoI/eccov4r5_OI_torch.py --out-dir OI_output_torch
    python scripts_TomsQoI/eccov4r5_OI_torch.py --lags 0 --kind pos --oi-iters 200 --no-plots    # quick test

The networks are, with --source saved-keras, the saved Keras models of --saved-models-dir, the weights of which are copied
into PyTorch models (keras_to_torch), so the OI is of the network of the notebooks. With --source train they are trained
with PyTorch as in eccov4r5_LRP_A1B0_torch.py.

The data, the experiment and the OI are in eccov4r5_common.py and eccov4r5_OI_common.py and are shared with the Keras script.
"""

import sys
from pathlib import Path

import numpy as np
# Also hides the GPU from TensorFlow, which is only needed here to load Keras models. The GPU is for PyTorch.
import eccov4r5_LRP_A1B0_torch as torch_script
import torch

# Append the src directory of the repository to sys.path, then import XAIRT
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from XAIRT import Torch_GradientDescent_useAutograd, torch_to_numpy

import eccov4r5_common as common
import eccov4r5_OI_common as oi_common

def optimal_input(model, desired_labels, init, eta, n_iters, print_freq):

    model.eval()
    x = torch.from_numpy(init.astype(np.float32))
    desired = torch.from_numpy(desired_labels)
    bce = torch.nn.BCELoss()

    print(f"Desired label : {desired_labels}")
    with torch.no_grad():
        print(f"Iter 0, Prediction {torch_to_numpy(model(x))}")

    for i in range(n_iters):
        grads = Torch_GradientDescent_useAutograd(model, x, desired, bce)
        x = x - eta*grads
        if (i+1) % print_freq == 0:
            with torch.no_grad():
                print(f"Iter {i+1}, Prediction {torch_to_numpy(model(x))}")

    return torch_to_numpy(x)[0]

if __name__ == "__main__":

    args = common.parse_args("torch", lambda p: oi_common.add_oi_args(p, ('saved-keras', 'train')))

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Running on GPU" if torch.cuda.is_available() else "Running on CPU")

    oi_common.run_oi("torch", args, torch_script.make_get_model(args), torch_script.predict, optimal_input)
