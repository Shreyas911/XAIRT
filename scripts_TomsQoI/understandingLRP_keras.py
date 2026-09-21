"""
LRP on a network with a single Dense layer with the Keras backend (innvestigate) next to the same relevance
computed by hand, converted from notebooks_TomsQoI/understandingLRP.ipynb

    python scripts_TomsQoI/understandingLRP_keras.py --out-dir simpleTests_output_keras
    python scripts_TomsQoI/understandingLRP_keras.py --epochs 2 --n-samples 500    # quick test

LRP-A1B0, WSquare, Bounded with the bounds (-20, 20), (0.4, 0.6) and (-0.4, 0.6) as well, Z and Epsilon. It is
LRP_manual_MWE_keras.py plus those bounds. The network is the one of simpleTests_LRP_keras.py, the setup and the
helpers are imported from there.

The notebook gave innvestigate (-0.4, 0.6) and computed (0.4, 0.6) by hand. It is not clear if that was a typo or a
check of whether the library needs different bounds to match the hand computation, so both pairs are run with the library
and by hand here, and can be compared. The inputs are in [0, 1], so neither pair contains them, and Bounded is not
meaningful for them, but the relevance still adds up to the output. The BoundedRule of innvestigate 2.1.0 uses
z = x.w - low.w+ - high.w- with the (low, high) it is given, which is the hand computation, so the same bounds
are expected to agree.

The bias, https://github.com/albermax/innvestigate/issues/327: in innvestigate 2.1.0 lrp.alpha_1_beta_0 gives the bias a share of the relevance, so
R_last - sum(a) is not 0 for it, see LRP_manual_MWE_keras.py, which is the minimal working example of that issue.

The Torch counterpart is understandingLRP_torch.py.
"""

import simpleTests_common as common
from simpleTests_LRP_keras import init, setup, print_analyzers, a1b0_analyzers, Z_EPSILON_ANALYZERS

# The bounds of LRP-Bounded besides the default (-1, 1), for the library and by hand
BOUNDS = ((-20, 20), (0.4, 0.6), (-0.4, 0.6))

if __name__ == "__main__":

    args = common.parse_args("keras", epochs = 100, n_samples = 10000)
    init(args)

    model_wo_softmax, x, W, b, pred_class, R_last = setup(args)

    print_analyzers(model_wo_softmax, x, R_last, a1b0_analyzers(bounds = BOUNDS) + Z_EPSILON_ANALYZERS)

    common.manual_lrp(x, W, b, pred_class, R_last, bounds = ((-1, 1),) + BOUNDS)
    common.manual_lrp_z_epsilon(x, W, b, pred_class, R_last)
