"""
LRP on a network with a single Linear layer with the PyTorch backend (Captum) next to the same relevance
computed by hand, the counterpart of LRP_manual_MWE_keras.py, which is converted from
notebooks_TomsQoI/LRP_manual_MWE.ipynb

    python scripts_TomsQoI/LRP_manual_MWE_torch.py --out-dir simpleTests_output_torch
    python scripts_TomsQoI/LRP_manual_MWE_torch.py --epochs 2 --n-samples 500    # quick test

This notebook is a subset of understandingLRP.ipynb, which is understandingLRP_torch.py: the same network and
the same methods, the latter only adds custom bounds for LRP-Bounded. Captum only has LRP-A1B0, Z and Epsilon, so
for WSquare and Bounded, and for the variants that ignore the bias, only the relevance computed by hand is printed.

The bias, https://github.com/albermax/innvestigate/issues/327: innvestigate gives the bias of LRP-A1B0 a share of the relevance, see LRP_manual_MWE_keras.py. How Captum
treats it was not checked, the last number of each line, R_last - sum(a), shows it, and which of the three hand computations of
A1B0 it agrees with.
"""

import simpleTests_common as common
from simpleTests_LRP_torch import init, setup, print_analyzers, A1B0_ANALYZERS, Z_EPSILON_ANALYZERS

if __name__ == "__main__":

    args = common.parse_args("torch", epochs = 100, n_samples = 10000)
    init(args)

    model_wo_softmax, x, W, b, pred_class, R_last = setup(args)

    print_analyzers(model_wo_softmax, x, R_last, A1B0_ANALYZERS + Z_EPSILON_ANALYZERS)

    common.manual_lrp(x, W, b, pred_class, R_last, bounds = ((-1, 1),))
    common.manual_lrp_z_epsilon(x, W, b, pred_class, R_last)
