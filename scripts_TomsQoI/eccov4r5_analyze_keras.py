"""
ECCOv4r5 predictions and accuracy of the saved networks with the Keras backend (TensorFlow), converted from
notebooks_TomsQoI/eccov4r5_classification-analyze-savedNN.ipynb

    python scripts_TomsQoI/eccov4r5_analyze_keras.py --out-dir Analyze_output_keras
    python scripts_TomsQoI/eccov4r5_analyze_keras.py --lags 0 --no-plots    # quick test

The networks are the saved Keras models of --saved-models-dir, as in the notebook, or with --source train they are
trained as in eccov4r5_LRP_A1B0_keras.py. Writes qoi_pred_keras.nc, that results_viz_qoi_pred.py plots, and
accuracy_f1_keras.nc, see eccov4r5_analyze_common.py, where the data and the analysis are, shared with
eccov4r5_analyze_torch.py, which does the same with PyTorch. The plots are not made here, --no-plots is
only there since the options are the ones of the other ECCOv4r5 scripts.
"""

import tensorflow as tf
import tensorflow.keras as keras

import eccov4r5_common as common
import eccov4r5_analyze_common as analyze_common
import eccov4r5_LRP_A1B0_keras as keras_script

if __name__ == "__main__":

    args = common.parse_args("keras", lambda p: common.add_model_args(p, ('saved', 'train')))

    ### https://stackoverflow.com/questions/36288235/how-to-get-stable-results-with-tensorflow-setting-random-seed ###
    ### https://keras.io/examples/keras_recipes/reproducibility_recipes/ ###
    keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()

    print("Running on GPU" if tf.config.list_physical_devices('GPU') else "Running on CPU")

    analyze_common.run_analyze("keras", args, keras_script.make_get_model(args), keras_script.predict)
