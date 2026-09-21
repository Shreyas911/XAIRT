"""
ECCOv4r5 predictions and accuracy of the saved networks with the PyTorch backend, the counterpart of
eccov4r5_analyze_keras.py, which is converted from notebooks_TomsQoI/eccov4r5_classification-analyze-savedNN.ipynb

    python scripts_TomsQoI/eccov4r5_analyze_torch.py --out-dir Analyze_output_torch
    python scripts_TomsQoI/eccov4r5_analyze_torch.py --lags 0 --no-plots    # quick test

The networks are, with --source saved-keras, the saved Keras models of --saved-models-dir, the weights of which are copied
into PyTorch models (keras_to_torch), so the predictions are those of the networks of the notebook. With --source train
they are trained with PyTorch as in eccov4r5_LRP_A1B0_torch.py. Writes qoi_pred_torch.nc, that results_viz_qoi_pred.py plots,
and accuracy_f1_torch.nc, see eccov4r5_analyze_common.py, where the data and the analysis are, shared with the Keras
script. The plots are not made here, --no-plots is only there since the options are the ones of the other ECCOv4r5 scripts.
"""

# Also hides the GPU from TensorFlow, which is only needed here to load Keras models. The GPU is for PyTorch.
import eccov4r5_LRP_A1B0_torch as torch_script
import torch

import eccov4r5_common as common
import eccov4r5_analyze_common as analyze_common

if __name__ == "__main__":

    args = common.parse_args("torch", lambda p: common.add_model_args(p, ('saved-keras', 'train')))

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Running on GPU" if torch.cuda.is_available() else "Running on CPU")

    analyze_common.run_analyze("torch", args, torch_script.make_get_model(args), torch_script.predict)
