# Preprint

A preprint of our work can be found [here]() (to be published soon).

# Abstract

In recent years neural networks (NN) have become a popular machine learning tool within geophysical research, enabled by ongoing algorithmic advances, increasing computer power, and a massive expansion in the availability of large, high-quality datasets. A successful application yields an algorithm that can skillfully classify or predict variations in a target quantity of interest (QoI) from input data not used for training. Whilst NNs are supporting significant advances in areas such as weather forecasting, subgrid-scale parameterization, and state estimation, they are inherently challenging to interpret. Recent work has sought to enhance explainability by evaluating key patterns within the training data informing QoI estimation. Examples of eXplainable AI (XAI) approaches include Backwards optimization (optimal input) and Layerwise Relevance Propagation (LRP). Due to their fundamentally empirical origins, however, relevance patterns (or ``heat maps'') inferred via LRP are still best-suited to formulate hypotheses for further investigation.  If NNs are detecting true dynamical connections, these heat maps should expose mechanistic pathways underpinning QoI variability, analogous with unambiguously dynamics-based sensitivity distributions derived from an adjoint model.

Here we pursue this comparison. By considering a simple case of estimating SST variability in the northeast Pacific from near-global historic SST, we show that NN (statistics-based) and adjoint (dynamics-based) methods both identify the importance of SST anomalies generated within the upstream eastern boundary and equatorial waveguides. Important discrepancies are also seen, however, with only the NN indicating relevance of SST anomalies in remote ocean basins on relatively short timescales. We show that in some cases, this remote relevance can be attributed to the existence of atmospheric bridges that are absent from the adjoint of the ocean-only model. In other cases, however, this relevance appears spurious in lacking physical explanation and likely arises from overexertion of the NN machinery in training on SST inputs alone, which the adjoint shows is not the only causal forcing. Our results demonstrate the benefit of pairing NNs and adjoint investigations for understanding and attributing ocean variability, and highlight the pitfalls of using NN-based assessments in isolation. 

# Dependencies

The Python dependencies needed to run the code, for both the Keras (`innvestigate`) and PyTorch (`captum`) backends, can be installed using the conda environment YAML file [env/XAIRT.yaml](env/XAIRT.yaml):

```
mamba env create -f env/XAIRT.yaml
conda activate XAIRT
```

On a machine without a GPU, conda-forge's GPU builds of TensorFlow need a CUDA driver to be faked:

```
CONDA_OVERRIDE_CUDA=11.2 mamba env create -f env/XAIRT.yaml
```

Building the environment on a login node of Lonestar6 can fail with `std::bad_alloc`, since the login nodes cap the virtual memory of a process. [env/create_env.slurm](env/create_env.slurm) builds it on a GPU node instead (`sbatch env/create_env.slurm`, from the repository root), and prints the versions of the key packages and whether TensorFlow and PyTorch see the GPU when it is done.

The environment is for x86_64 Linux (for example Lonestar6). It pins TensorFlow 2.9.1 because `innvestigate` needs `tensorflow<2.15` and Python 3.10 is the last version TF 2.9.1 supports. TF 2.9.1 has no ARM wheel, so this file does not work on ARM machines such as Vista. The reasons for each pin are in the comments at the top of the YAML file.

# Code structure

The library is in [src/XAIRT](src/XAIRT). Every backend specific class comes in a Keras and a PyTorch version with the same interface.

[src/XAIRT/Trainer.py](src/XAIRT/Trainer.py): `TrainKerasFullyConnectedNN` and `TrainTorchFullyConnectedNN` train the same fully connected network from the same layer description (sizes, activations, L1/L2 regularization, constraints), loss, optimizer, learning rate schedule and class weights, and keep the model with the best validation loss. `TrainLR` fits a linear regression.

[src/XAIRT/XAI.py](src/XAIRT/XAI.py): `XAIKeras` (`innvestigate`) and `XAITorch` (`captum`) explain a network with the same method dictionary, e.g. `dict(name='lrp.alpha_1_beta_0', optParams={})`, and normalize the relevance in the same way. `XLR` explains a linear regression. While we use them for a classification task, these classes can ideally also handle regression-based XAI against various benchmarks, as pioneered by [Letzgus et. al 2022](10.1109/msp.2022.3153277).

[src/XAIRT/utils.py](src/XAIRT/utils.py): gradient of the loss w.r.t. the input for optimal input (OI) iterations (`Keras_GradientDescent_useGradientTape`, `Torch_GradientDescent_useAutograd`), `keras_to_torch` that copies the weights of a Keras network into a PyTorch one so that both libraries can explain exactly the same network, `model_wo_softmax_torch` (Captum LRP does not support a final softmax), the F1 score metric and the correlation.

# Scripts

The scripts are in [scripts_TomsQoI](scripts_TomsQoI). They are converted from the notebooks (see [Legacy notebooks](#legacy-notebooks)), and they replace them. Where a script needs a backend there is a `_keras` and a `_torch` version that run the identical experiment, and a `_common` module with everything that has to be the same in both: the data, the network, the loss, the learning rate schedule, the splits and the analysis. Scripts that only handle data or plots have a single version, and say so in their docstring. Every script has `--help`, and a quick test command in its docstring.

> **Read [HANDOFF.md](HANDOFF.md) first:** what has not been verified, what is open, and what to run first.
>
> **Status:** the scripts have been converted from the notebooks and compile, and the numpy parts (the relevance computed by hand, the correlation, the lag alignment) were tested on their own, but none has been run end to end yet. Update this note as they are tested.

## The three cases

Wherever a network is needed, one can look at:

1. a network trained with Keras and explained with Keras (`innvestigate`), the `_keras` scripts,
2. a network trained with PyTorch and explained with PyTorch (`captum`), the `_torch` scripts with `--source train`,
3. a network trained with Keras, its weights copied into PyTorch and explained with PyTorch, the `_torch` scripts with `--source saved-keras`, which loads the saved Keras models of `--saved-models-dir`.

Comparing 1 and 3 isolates the difference between the two XAI libraries, since it is the same network, whereas 2 also differs by training.

## ECCOv4r5, the research workflow

The data, the paths (`--r4-dir`, `--r5-dir`) and the lags (`--lags`) are the same for all of these, see `eccov4r5_common.py`. The usual order is:

| Step | Script | What it does |
|---|---|---|
| 1. Data | [eccov4r5_dataReader.py](scripts_TomsQoI/eccov4r5_dataReader.py) | `--build-sst` reads the daily SST files and writes `SST_all.nc`, the input of everything else. Prints the statistics of the new anomalies that justify LRP-Bounded for the first layer, and plots them and the correlation maps. |
| 2. Train and LRP | [eccov4r5_LRP_A1B0_keras.py](scripts_TomsQoI/eccov4r5_LRP_A1B0_keras.py), [eccov4r5_LRP_A1B0_torch.py](scripts_TomsQoI/eccov4r5_LRP_A1B0_torch.py) | Trains a network for each lag, reports accuracy and F1, and explains the correctly predicted samples of each class with LRP-A1B0. Writes the metrics, the mean relevance maps as NetCDF and the figures. Keras also runs LRP-A1B0-B, with the `Bounded` input layer rule of the notebook, which Captum does not have. |
| 3. Compare libraries | [eccov4r5_LRP_A1B0_compare.py](scripts_TomsQoI/eccov4r5_LRP_A1B0_compare.py) | Case 1 against case 3: trains once with Keras, copies the weights into PyTorch and runs LRP-A1B0 and LRP-A1B0-IB with both libraries on the same samples. Reports the difference of the predictions (about 1e-6 if the weights were copied right) and the Pearson and cosine agreement of the relevance. The libraries are expected to differ in the first layer, since the anomalies have both signs: innvestigate splits the input into its positive and negative parts, Captum does not. They should agree for inputs of one sign, e.g. after a ReLU. |
| 4. Optimal input | [eccov4r5_OI_keras.py](scripts_TomsQoI/eccov4r5_OI_keras.py), [eccov4r5_OI_torch.py](scripts_TomsQoI/eccov4r5_OI_torch.py) | Gradient descent on the input, starting from the mean of the correctly predicted samples of a class, until the network gives that class. `--kind pos`, `neg` or `both`. The networks are the saved Keras models (`--source saved` / `saved-keras`), or trained with `--source train`. |
| 5. Analyze | [eccov4r5_analyze_keras.py](scripts_TomsQoI/eccov4r5_analyze_keras.py), [eccov4r5_analyze_torch.py](scripts_TomsQoI/eccov4r5_analyze_torch.py) | Predictions of the network of each lag on the whole series, the training part and the test part, against the true class, with the accuracy and F1 of each. Writes `qoi_pred_*.nc` and `accuracy_f1_*.nc`. Same `--source` options as the OI. |
| 6. Plot | [eccov4r5_results_viz_new.py](scripts_TomsQoI/eccov4r5_results_viz_new.py), [eccov4r5_results_viz.py](scripts_TomsQoI/eccov4r5_results_viz.py), [results_viz_qoi_pred.py](scripts_TomsQoI/results_viz_qoi_pred.py) | Plots of the LRP, OI, composite and correlation maps of the results by lag (`_new` for the result files of the scripts above, the other for the older result files, including the `_small` variants), and of where the predicted class is wrong. No backend, so a single version. |

The shared modules are [eccov4r5_common.py](scripts_TomsQoI/eccov4r5_common.py) (data, experiment, analysis, plots), [eccov4r5_OI_common.py](scripts_TomsQoI/eccov4r5_OI_common.py), [eccov4r5_analyze_common.py](scripts_TomsQoI/eccov4r5_analyze_common.py) and [eccov4r5_viz_common.py](scripts_TomsQoI/eccov4r5_viz_common.py).

The anomalies are the new ones: each calendar day is delinearized separately over the years. The old ones (one linear trend for the whole record) are not used anywhere.

## Fake data

These need no ECCO data, so they are the quickest way to check an installation.

| Script | What it does |
|---|---|
| [fake_data_LRP_A1B0_compare.py](scripts_TomsQoI/fake_data_LRP_A1B0_compare.py) | The three cases above on a small fake classification problem with a known answer, for LRP-A1B0 and LRP-A1B0-IB. Reports the agreement of the relevance of each pair, and the share of the relevance on the informative features. `--positive-inputs` makes every layer's input non-negative, where the two libraries are expected to agree to float precision, so a difference then is not the input sign. |
| [simpleTests_LRP_keras.py](scripts_TomsQoI/simpleTests_LRP_keras.py), [simpleTests_LRP_torch.py](scripts_TomsQoI/simpleTests_LRP_torch.py) | LRP on a network with a single layer of 2 inputs, next to the same relevance computed by hand, and LRP of a regression network next to a linear regression (XLR). |
| [LRP_manual_MWE_keras.py](scripts_TomsQoI/LRP_manual_MWE_keras.py), [LRP_manual_MWE_torch.py](scripts_TomsQoI/LRP_manual_MWE_torch.py) | LRP-A1B0, WSquare, Bounded, Z and Epsilon of that network, next to the relevance by hand. This is the minimal working example of the `innvestigate` issue [#327](https://github.com/albermax/innvestigate/issues/327): `lrp.alpha_1_beta_0` keeps the bias as an input neuron that takes a share of the relevance, so the relevance of the inputs does not add up to the output, while the variants that ignore the bias do. A subset of the next one. |
| [understandingLRP_keras.py](scripts_TomsQoI/understandingLRP_keras.py), [understandingLRP_torch.py](scripts_TomsQoI/understandingLRP_torch.py) | The same with custom bounds for LRP-Bounded. These illustrate how `innvestigate` computes each rule. We suspect presence of bugs (see the issue above), and we chose to use only those LRP methods that we were convinced were free of bugs and conservative. Captum only has A1B0, Z and Epsilon, so only those are run there, the rest is the relevance by hand. |
| [simpleTests_OI_keras.py](scripts_TomsQoI/simpleTests_OI_keras.py), [simpleTests_OI_torch.py](scripts_TomsQoI/simpleTests_OI_torch.py) | Optimal input on a two-input classification problem, with plain gradient descent and with an optimizer (Adam, SGD) on the input. Keras runs with eager execution, which is needed for the optimizers but not possible with `innvestigate`, so the OI and the LRP scripts are separate. |

The common module is [simpleTests_common.py](scripts_TomsQoI/simpleTests_common.py).

## Files that are not in the repository

`*.h5` files are ignored by git, and so are the results. The scripts expect the saved Keras models `model{lag}_noL1.h5` in `--saved-models-dir` and the result files in `--results-dir`, both `LRP_output_forHelen/...` by default, as in the notebooks.

# Legacy notebooks

> **The notebooks are legacy and will be removed** once the scripts above are up and running. They are kept, in [notebooks_TomsQoI](notebooks_TomsQoI), for reference until then. They still use function names that no longer exist in `src/XAIRT` (`TrainFullyConnectedNN`, `XAIR`, `GradientDescent_useGradientTape`, ...), so they do not run as they are. Please use the scripts.

| Notebook | Replaced by |
|---|---|
| [LRP_manual_MWE.ipynb](notebooks_TomsQoI/LRP_manual_MWE.ipynb) | `LRP_manual_MWE_{keras,torch}.py` |
| [understandingLRP.ipynb](notebooks_TomsQoI/understandingLRP.ipynb) | `understandingLRP_{keras,torch}.py` |
| [simpleTests.ipynb](notebooks_TomsQoI/simpleTests.ipynb) | `simpleTests_LRP_{keras,torch}.py` (the LRP and the regression), `simpleTests_OI_{keras,torch}.py` (the OI) |
| [simpleTests_eagerExecution.ipynb](notebooks_TomsQoI/simpleTests_eagerExecution.ipynb) | `simpleTests_OI_{keras,torch}.py` |
| [eccov4r5_dataReader-newAnomalies.ipynb](notebooks_TomsQoI/eccov4r5_dataReader-newAnomalies.ipynb) | `eccov4r5_dataReader.py` |
| [eccov4r5_classification-LRP-A1B0-IB-B-newAnomalies-shuffleVal-reweight.ipynb](notebooks_TomsQoI/eccov4r5_classification-LRP-A1B0-IB-B-newAnomalies-shuffleVal-reweight.ipynb) | `eccov4r5_LRP_A1B0_{keras,torch,compare}.py` |
| [eccov4r5_classification-OI-pos-newAnomalies-shuffleVal-reweight-avgInit-savedNN.ipynb](notebooks_TomsQoI/eccov4r5_classification-OI-pos-newAnomalies-shuffleVal-reweight-avgInit-savedNN.ipynb) | `eccov4r5_OI_{keras,torch}.py --kind pos` |
| [eccov4r5_classification-OI-neg-newAnomalies-shuffleVal-reweight-avgInit-savedNN.ipynb](notebooks_TomsQoI/eccov4r5_classification-OI-neg-newAnomalies-shuffleVal-reweight-avgInit-savedNN.ipynb) | `eccov4r5_OI_{keras,torch}.py --kind neg` |
| [eccov4r5_classification-analyze-savedNN.ipynb](notebooks_TomsQoI/eccov4r5_classification-analyze-savedNN.ipynb) | `eccov4r5_analyze_{keras,torch}.py` |
| [eccov4r5_results_viz_new.ipynb](notebooks_TomsQoI/eccov4r5_results_viz_new.ipynb) | `eccov4r5_results_viz_new.py` |
| [eccov4r5_results_viz.ipynb](notebooks_TomsQoI/eccov4r5_results_viz.ipynb) | `eccov4r5_results_viz.py` |
| [results_viz_qoi_pred.ipynb](notebooks_TomsQoI/results_viz_qoi_pred.ipynb) | `results_viz_qoi_pred.py` |

Not carried over into the scripts: the plots of the raw SST (first day, mean, the objective function point) and the comparison of the old and new anomalies of the data reader notebook, the animation of the results viz notebook (commented out there), and two bugs of `eccov4r5_results_viz_new.ipynb` that are fixed in the script (the composite figures used a stale lag index, and the OI negative figure was titled OI-pos).
