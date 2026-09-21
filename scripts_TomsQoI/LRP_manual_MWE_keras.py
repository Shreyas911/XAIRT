"""
LRP on a network with a single Dense layer with the Keras backend (innvestigate) next to the same relevance
computed by hand, converted from notebooks_TomsQoI/LRP_manual_MWE.ipynb

    python scripts_TomsQoI/LRP_manual_MWE_keras.py --out-dir simpleTests_output_keras
    python scripts_TomsQoI/LRP_manual_MWE_keras.py --epochs 2 --n-samples 500    # quick test

LRP-A1B0, WSquare, Bounded, Z and Epsilon. This notebook is a subset of understandingLRP.ipynb, which is
understandingLRP_keras.py: the same network and the same methods, the latter only adds custom bounds for LRP-Bounded.
The network is the one of simpleTests_LRP_keras.py, the setup and the helpers are imported from there.

The bias, https://github.com/albermax/innvestigate/issues/327, which this is the minimal working example of: in innvestigate 2.1.0 lrp.alpha_1_beta_0 keeps the bias as
a constant input neuron, so it takes its share of the relevance, and the last number of each line, R_last - sum(a), is that share.
It is 0 for the variants that ignore the bias (_IB, WSquare, Bounded). Of the three hand computations of A1B0 the one that puts one
bias in the denominator and none in the numerator is expected to agree with the library, and the one with 2*bias, split between
the two inputs, is the one that conserves the relevance.

The Torch counterpart is LRP_manual_MWE_torch.py.
"""

import simpleTests_common as common
from simpleTests_LRP_keras import init, setup, print_analyzers, a1b0_analyzers, Z_EPSILON_ANALYZERS

if __name__ == "__main__":

    args = common.parse_args("keras", epochs = 100, n_samples = 10000)
    init(args)

    model_wo_softmax, x, W, b, pred_class, R_last = setup(args)

    print_analyzers(model_wo_softmax, x, R_last, a1b0_analyzers() + Z_EPSILON_ANALYZERS)

    common.manual_lrp(x, W, b, pred_class, R_last, bounds = ((-1, 1),))
    common.manual_lrp_z_epsilon(x, W, b, pred_class, R_last)
