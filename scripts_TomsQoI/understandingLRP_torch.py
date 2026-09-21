"""
LRP on a network with a single Linear layer with the PyTorch backend (Captum) next to the same relevance
computed by hand, the counterpart of understandingLRP_keras.py, which is converted from
notebooks_TomsQoI/understandingLRP.ipynb

    python scripts_TomsQoI/understandingLRP_torch.py --out-dir simpleTests_output_torch
    python scripts_TomsQoI/understandingLRP_torch.py --epochs 2 --n-samples 500    # quick test

It is LRP_manual_MWE_torch.py plus the custom bounds of LRP-Bounded, which only exist by hand here: Captum only has
LRP-A1B0, Z and Epsilon, so for WSquare and Bounded, and for the variants that ignore the bias, only the relevance
computed by hand is printed.

The bounds by hand are (-1, 1), (-20, 20), (0.4, 0.6) and (-0.4, 0.6). The notebook gave innvestigate (-0.4, 0.6) and computed
(0.4, 0.6) by hand, and it is not clear if that was a typo or a check of whether the library needs different bounds to match,
so both pairs are computed. The Keras script runs both with the library too. The inputs are in [0, 1], so neither pair contains
them, and Bounded is not meaningful for them, but the relevance still adds up to the output.

The bias, https://github.com/albermax/innvestigate/issues/327: innvestigate gives the bias of LRP-A1B0 a share of the relevance, see LRP_manual_MWE_keras.py, which is the
minimal working example of that issue. How Captum treats it was not checked, R_last - sum(a) shows it.
"""

import simpleTests_common as common
from simpleTests_LRP_torch import init, setup, print_analyzers, A1B0_ANALYZERS, Z_EPSILON_ANALYZERS

if __name__ == "__main__":

    args = common.parse_args("torch", epochs = 100, n_samples = 10000)
    init(args)

    model_wo_softmax, x, W, b, pred_class, R_last = setup(args)

    print_analyzers(model_wo_softmax, x, R_last, A1B0_ANALYZERS + Z_EPSILON_ANALYZERS)

    common.manual_lrp(x, W, b, pred_class, R_last, bounds = ((-1, 1), (-20, 20), (0.4, 0.6), (-0.4, 0.6)))
    common.manual_lrp_z_epsilon(x, W, b, pred_class, R_last)
