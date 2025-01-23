# Preprint

A preprint of our work can be found [here]() (to be published soon).

# Abstract

In recent years neural networks (NN) have become a popular machine learning tool within geophysical research, enabled by ongoing algorithmic advances, increasing computer power, and a massive expansion in the availability of large, high-quality datasets. A successful application yields an algorithm that can skillfully classify or predict variations in a target quantity of interest (QoI) from input data not used for training. Whilst NNs are supporting significant advances in areas such as weather forecasting, subgrid-scale parameterization, and state estimation, they are inherently challenging to interpret. Recent work has sought to enhance explainability by evaluating key patterns within the training data informing QoI estimation. Examples of eXplainable AI (XAI) approaches include Backwards optimization (optimal input) and Layerwise Relevance Propagation (LRP). Due to their fundamentally empirical origins, however, relevance patterns (or ``heat maps'') inferred via LRP are still best-suited to formulate hypotheses for further investigation.  If NNs are detecting true dynamical connections, these heat maps should expose mechanistic pathways underpinning QoI variability, analogous with unambiguously dynamics-based sensitivity distributions derived from an adjoint model.

Here we pursue this comparison. By considering a simple case of estimating SST variability in the northeast Pacific from near-global historic SST, we show that NN (statistics-based) and adjoint (dynamics-based) methods both identify the importance of SST anomalies generated within the upstream eastern boundary and equatorial waveguides. Important discrepancies are also seen, however, with only the NN indicating relevance of SST anomalies in remote ocean basins on relatively short timescales. We show that in some cases, this remote relevance can be attributed to the existence of atmospheric bridges that are absent from the adjoint of the ocean-only model. In other cases, however, this relevance appears spurious in lacking physical explanation and likely arises from overexertion of the NN machinery in training on SST inputs alone, which the adjoint shows is not the only causal forcing. Our results demonstrate the benefit of pairing NNs and adjoint investigations for understanding and attributing ocean variability, and highlight the pitfalls of using NN-based assessments in isolation. 

# Dependencies

The Python dependencies needed to run the code can be installed using the conda environment YAML file [env/py310_LRP.yaml](env/py310_LRP.yaml). 

# Code Structure

Not all of the code is relevant to the final research presented in the paper. We highlight the relevant parts of the code here and also leave the rest of the code intact for reference/future work.

[src/XAIRT/backend/graph.py](src/XAIRT/backend/graph.py): Contains definitions of useful functions to handle the tensorflow computational graph such as checking if a certain layer exists within a NN, a function for gradient descent useful for Optimal Input (OI) when eager execution has to be disabled (a requirement for using the `innvestigate` package properly), etc.

[src/XAIRT/backend/metrics.py](src/XAIRT/backend/metrics.py): Contains definitions to custom performance metrics, such as F1-score.

[src/XAIRT/backend/types.py](src/XAIRT/backend/types.py): Contains definitions of types and custom types for typing hints throughout the code.

[src/XAIRT/model/Trainer.py](src/XAIRT/model/Trainer.py): Contains class definitions to train neural networks, especially relevant is the class `TrainFullyConnectedNN`.

[src/XAIRT/model/XAI.py](src/XAIRT/model/XAI.py): Contains class definitions for interpreting neural networks, especially relevant is the class `XAIR`. While we use it for a classification task, this class can ideally also handle regression-based XAI (XAIR) against various benchmarks, as pioneered by [Letzgus et. al 2022](10.1109/msp.2022.3153277).

[src/XAIRT/utils/stats.py](src/XAIRT/utils/stats.py): Utility functions for some statistics.

[src/XAIRT/utils/visualizations.py](src/XAIRT/utils/visualizations.py): Utility functions for visualizations.

# Jupyter notebooks

This is where we get the final results and visualize them.

* Notebooks that illustrate examples of us trying to understand the `innvestigate` package. We suspect presence of bugs and the git issue has been already raised [here](https://github.com/albermax/innvestigate/issues/327). We chose to use only those LRP methods that we were convinced were free of bugs and conservative.

[examples/LRP_manual_MWE.ipynb](examples/LRP_manual_MWE.ipynb)

[examples/understandingLRP.ipynb](examples/understandingLRP.ipynb)

* Notebooks where we are understanding how to do optimal input iterations with and without eager execution enabled. While it would be really great to have eager execution, it was not easy to use it with the `innvestigate` package present in the environment.

[examples/simpleTests.ipynb](examples/simpleTests.ipynb)

[examples/simpleTests_eagerExecution.ipynb](examples/simpleTests_eagerExecution.ipynb)

* Notebooks for visualizing data and results.

[examples/eccov4r5_dataReader-newAnomalies.ipynb](examples/eccov4r5_dataReader-newAnomalies.ipynb)

[examples/eccov4r5_results_viz_new.ipynb](examples/eccov4r5_results_viz_new.ipynb)

[examples/eccov4r5_results_viz.ipynb](examples/eccov4r5_results_viz.ipynb)

[examples/eccov4r5_classification-analyze-savedNN.ipynb](examples/eccov4r5_classification-analyze-savedNN.ipynb)

[examples/results_viz_qoi_pred.ipynb](examples/results_viz_qoi_pred.ipynb)

* Notebooks for inferring optimal inputs.

[examples/eccov4r5_classification-OI-pos-newAnomalies-shuffleVal-reweight-avgInit-savedNN.ipynb](examples/eccov4r5_classification-OI-pos-newAnomalies-shuffleVal-reweight-avgInit-savedNN.ipynb)

[examples/eccov4r5_classification-OI-neg-newAnomalies-shuffleVal-reweight-avgInit-savedNN.ipynb](examples/eccov4r5_classification-OI-neg-newAnomalies-shuffleVal-reweight-avgInit-savedNN.ipynb)

* Notebooks for inferring optimal inputs.

[examples/eccov4r5_classification-LRP-A1B0-IB-B-newAnomalies-shuffleVal-reweight.ipynb](examples/eccov4r5_classification-LRP-A1B0-IB-B-newAnomalies-shuffleVal-reweight.ipynb)

