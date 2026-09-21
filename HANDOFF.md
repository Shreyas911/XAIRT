# Handoff: state of the notebook-to-script conversion

Written at the end of a long working session so that the next one does not have to rediscover it. It says what was done, what has
**not** been verified, what is open, and what to run first. Read the README for what the scripts do, this file for what is missing.
Delete it once the items below are resolved.

## In one paragraph

All 12 notebooks of `notebooks_TomsQoI/` have a script counterpart in `scripts_TomsQoI/` (Keras and Torch versions where a backend
is needed, one version where not), documented in the README. **None of the scripts has been run.** No environment on the machine could
import the stack when they were written, so they were checked only by compiling them and by testing their numpy parts. Expect a
first round of small runtime errors (type checks with `beartype`/`jaxtyping`, argument mismatches, dtype issues) when they meet the
real libraries.

## Update at the end of the session: the environment works

`pip install torch==2.8.0` was run in the existing `XAIRT` env (the fix of the NCCL clash described in step 1 below), and it is done. Checked on the login node:
`import tensorflow, torch` and `import XAIRT` work, torch is `2.8.0+cu128`, and numpy 1.26.4 / protobuf 3.20.3 / keras 2.9.0 / TensorFlow 2.9.1 / captum 0.9.0 / innvestigate 2.1.2
are unchanged. The A100 nodes have driver 570.195.03, which supports CUDA 12.8 (not 13). **Not checked yet:** `torch.cuda.is_available()` and the TensorFlow GPUs on a GPU node
(run `sbatch env/create_env.slurm`, which now only checks an existing env).

Scripts run so far (first real runs, small settings, output in the scratchpad, not kept), all on the login node, where TensorFlow needs
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=` or it dies with
"Thread ... creation via pthread_create() failed" (the login node limits threads and memory; a compute node does not need this):
- `LRP_manual_MWE_keras.py --epochs 2 --n-samples 500`: runs. innvestigate agrees with the hand computation wherever it should: WSquare and Bounded (-1,1) match to 1e-7, `lrp.alpha_1_beta_0` equals
  the "one bias in the denominator, none in the numerator" variant with the residual `R_last - sum(a)` equal to the bias (0.144), `_IB` equals the "without bias" variant with residual 0.
  This is issue #327 reproduced, and confirms the reading of `BoundedRule` and `AlphaBetaRule`.
- `LRP_manual_MWE_torch.py --epochs 2 --n-samples 500`: runs, and Captum's A1B0, Z and epsilon match the hand computation. But the predicted class had bias 0 (NonNeg bias), so it says nothing yet about how Captum treats the bias.
- `fake_data_LRP_A1B0_compare.py --positive-inputs --epochs 20 --n-samples 1500 --n-explain 30`: runs (with the `XAITorch` fix and the IB mapping). Keras vs Keras-network-in-Torch: Pearson and cosine **1.0000** for both A1B0 and
  A1B0-IB, prediction difference 3.6e-07. As predicted for non-negative inputs. `keras` vs `torch` is about 0.66, which is expected since those are two separately trained networks.
  **One small run, one seed.** Still to do: the same without `--positive-inputs` (the hypothesis says the first layer should then differ), and the `XAITorch` regression test below.

## First things to do, in order

1. **Build and check the environment.** `sbatch env/create_env.slurm` (from the repo root) on a GPU node; if the env exists it only checks it.
   **Status when this was written:** the env was built once (job 3458672: numpy 1.26.4, protobuf 3.20.3, keras 2.9.0, TensorFlow 2.9.1, captum 0.9.0,
   innvestigate 2.1.2 by pip; `innvestigate.__version__` printed 2.0.1, probably a stale string, check with `pip show`), but **`import torch` after `import tensorflow`
   failed** with `undefined symbol: ncclCommResume`, so `import XAIRT` and every script failed. Cause: the unpinned pip torch was 2.14+cu130, which bundles NCCL 2.30,
   while conda has NCCL 2.27.3, and both are `libnccl.so.2`, so torch gets the conda one when TensorFlow loads first. Torch first works, TensorFlow first does not (reproduced on the
   login node with `OPENBLAS_NUM_THREADS=1`). The YAML now pins `torch==2.8.0`, which needs exactly NCCL 2.27.3 and is a CUDA 12.8 build. **This fix was not run**: fix the existing env in place with
   `mamba activate XAIRT; export PIP_CACHE_DIR=/work/07665/shrey911/ls6/pip_cache; pip install torch==2.8.0` (leaves the unused `nvidia-*-cu13` packages behind, harmless), then `sbatch env/create_env.slurm`
   to check it, and read the `Driver:` and `torch cuda:` lines: the earlier torch was CUDA 13, whether the A100 nodes' driver supports it, and CUDA 12.8, was never checked, and a torch that cannot see the GPU falls back to the CPU silently. The login node cannot do it: it caps the
   virtual memory of a process at 8 GB (`ulimit -v`), which makes mamba fail with `std::bad_alloc`. The home directory has a 10 GB
   quota that pip's cache filled once, so the script sends the pip cache to `/work` (`PIP_CACHE_DIR`). Check the version list at the end of
   the `.out` file: numpy 1.x, protobuf 3.20.x, keras 2.9.x and TensorFlow 2.9.1 are expected. The old `py310_LRP` env is broken (numpy 2,
   protobuf 7 and keras 3, all installed by pip after conda) and must not be used. The YAML no longer pins numpy, protobuf, cudatoolkit, cudnn or
   nccl, because the conda-forge TensorFlow build constrains them, and `protobuf<3.20` would conflict with its `libprotobuf 3.20`.
2. **Fake-data checks, no ECCO data needed**, in this order:
   - `python scripts_TomsQoI/fake_data_LRP_A1B0_compare.py --positive-inputs --epochs 5 --n-explain 20`
   - the same without `--positive-inputs`
   - `python scripts_TomsQoI/simpleTests_LRP_keras.py --epochs 2 --n-samples 500` and `..._torch.py`, then `LRP_manual_MWE_*`, `understandingLRP_*`, `simpleTests_OI_*`
     (`--epochs 1 --n-samples 20000` for the OI ones)
3. **Real-data scripts** on one lag: `--lags 0 --epochs 2`. Order: `eccov4r5_dataReader.py` (`--build-sst` only if `SST_all.nc` does not exist),
   `eccov4r5_LRP_A1B0_{keras,torch,compare}.py`, `eccov4r5_OI_*` and `eccov4r5_analyze_*` (both need saved models or `--source train`), then the viz scripts.
4. **Regression test for the `XAITorch` fix**, see below.

## The open hypothesis: do Captum and innvestigate agree?

This is one of the research hypotheses, and the scripts were built to test it. What is known comes **from reading source, not from running
it**: innvestigate 2.1.0 (the version in the old env, the YAML pins 2.1.2, which was not read) and Captum 0.9.0 (the version the env build installs).

- **The bias.** Both put the raw bias in the denominator of A1B0 and let the bias keep a share of the relevance, so the relevance on the
  inputs does not add up to the output (`R_last - sum(a)` is the bias share). This is what innvestigate issue
  [#327](https://github.com/albermax/innvestigate/issues/327) is about (its minimal working example is `LRP_manual_MWE`). Not a difference between the
  libraries. Whether it is a bug or the usual convention of treating the bias as a neuron that absorbs relevance is arguable. The variants that
  ignore the bias (`lrp.alpha_1_beta_0_IB`, WSquare, Bounded) are conservative by construction, and the `2*bias` split of the notebook is one
  arbitrary way (equal split over the inputs) to conserve. The real notebook uses `LRP-A1B0-IB-B`, so it is on the conservative side.
- **The input sign.** innvestigate uses `w+` on the positive part of the input and `w-` on the negative part. Captum clamps the weights to `w+` and
  applies them to the input as it is. Same for inputs >= 0 (hidden layers after a ReLU), different for inputs of both signs. The SST anomalies
  have both signs, so the **first layer is expected to differ**, for A1B0 and for A1B0-IB. Which library is "right" is not clear:
  Captum documents the rule for lower layers, i.e. post-ReLU inputs.

The scripts test it in both directions, and each outcome means something:

| Run | If they agree | If they disagree |
|---|---|---|
| fake data, `--positive-inputs` | expected | something other than the input sign differs, the reading above is incomplete |
| fake data, signed inputs (default) | the reading of the input split is wrong | expected, and the first layer is the cause |
| A1B0 vs A1B0-IB | the bias is not a factor | the gap is the bias effect |

`eccov4r5_LRP_A1B0_compare.py` does the same on the ECCO data (signed inputs, so disagreement is expected there).

## A bug found and fixed, and what it contaminates

`XAITorch` (in `src/XAIRT/XAI.py`) attached the LRP rule to the model once and then called Captum's `attribute()` for every sample. Captum
deletes the rules of the model at the end of **every** `attribute()` call, and a layer without a rule gets the default epsilon rule. So only
the first sample got A1B0, all the others were explained with (roughly) LRP-Z. Fixed by attaching the rules before every sample
(`_attach_lrp_rules`). **Any Torch LRP-A1B0 result computed with the old `XAITorch`, and any Keras-vs-Torch comparison based on it, is wrong and
cannot be evidence for or against the hypothesis above.** LRP-Z and epsilon were not affected.

This was found by reading Captum's `lrp.py`; it was **not reproduced**, because no environment had torch. To test the fix: on a small network and a few samples,
compare `XAITorch(...).quick_analyze()` with the relevance of each sample explained by a fresh `XAITorch` on that sample alone. They must be identical for
`lrp.alpha_1_beta_0`. Also worth confirming that Captum restores the bias after `lrp.alpha_1_beta_0_IB` (it does `load_state_dict` of the original state in `finally`).

## What changed relative to the notebooks (so that nobody is surprised)

- The notebooks call names that no longer exist (`TrainFullyConnectedNN`, `XAIR`, `GradientDescent_useGradientTape`, `TrainOI_useGradientTape`); the scripts use the current API.
- **Anomalies:** the viz notebooks used the old `anomalize` for the correlation maps, the LRP and OI results are of the new anomalies, and the new ones are the correct ones.
  The scripts use the new ones everywhere, and the old function was removed.
- Fake data of the simpleTests scripts is drawn with a seeded generator (the notebooks used the global numpy state), so the numbers differ from the notebooks.
- In the LRP fake data, class 0 (`x1 + 2 x2 >= 3`) is empty, since the sum is at most 3, so the network always predicts class 1 (hence `pred_class = 1` in the notebook).
  Kept as in the notebook, with a comment.
- `pred_class` is the argmax in all scripts (the notebook hard-coded 1 in one). The Adam demo recomputes the gradient every step (the notebook computed it once).
  The OI demo uses the trained network (the notebook cloned it, which does not copy the weights).
- `understandingLRP` gave innvestigate the bounds (-0.4, 0.6) and computed (0.4, 0.6) by hand. Unclear if it was a typo or a probe of the library, so both pairs
  are now run with the library and by hand. innvestigate 2.1.0's `BoundedRule` computes `z = x.w - low.w+ - high.w-` with the tuple as given, i.e. the hand formula, so the same bounds
  are expected to agree. The inputs are in [0,1], so neither pair contains them, and Bounded is not meaningful for them.
- `eccov4r5_results_viz_new.ipynb` bugs fixed in the script: the two composite figures used a stale lag index (every map was the same lag), and the OI negative figure was titled "OI-pos".
- Not converted: the plots of the raw SST and the old-vs-new anomalies comparison of the data reader notebook, the (commented out) animation of the viz notebook.
- `eccov4r5_common.py`: `load_data` is now a wrapper of `load_anomalies` + `make_qoi` (same result), and there are new `add_model_args`, `make_context`, `train_on_lag`. The LRP scripts got `make_get_model`.

## Data and paths

- `examples_TomsQoI/` is untracked and stale according to the owner, but it holds the **only copies** of the saved Keras models (`*.h5` is gitignored) and of
  `LRP_A1B0_IB_B_newAnomalies_shuffleVal_reweight.nc`. The scripts default to `LRP_output_forHelen/saved_models` and `LRP_output_forHelen` (the notebooks' path,
  relative to where the script is run), so copy the files there before deleting anything.
- Not in this checkout, so their viz panels are skipped with a message: the OI result files, and the older `LRP_A1B0*.nc` / `OI_*small*.nc` of `eccov4r5_results_viz.py`.
- `qoi_pred.nc`, which `results_viz_qoi_pred.py` plots, is written by `eccov4r5_analyze_{keras,torch}.py` (as `qoi_pred_{backend}.nc`, use `--file`).
- `eccov4r5_dataReader.py --build-sst` (xmitgcm, the mds files) was converted from code that was **commented out** in the notebook and has never been run in this form.

## Things most likely to break on the first run

- `beartype`/`jaxtyping` checks in `Trainer.py`/`XAI.py`: the fake data is float64 in places, y of `TrainLR` must be 1-D, samples given to `XAITorch` must be 2-D floats.
- innvestigate 2.1.2 (pinned) against the 2.1.0 that was read: rule names (`lrp.alpha_1_beta_0_IB`), `input_layer_rule` tuples, `bias=False` for epsilon.
- The eager-mode OI scripts call `Keras_GradientDescent_useGradientTape` (a `tf.function` with a `beartype`) with a fresh loss object per lag, which retraces. Probably fine, slow if not.
- `eccov4r5_OI_*`: the defaults `--eta 0.9999 --oi-iters 10000` are the notebook's values; 10000 iterations x 9 lags x 2 kinds is long.
- `keras_to_torch` on the saved `.h5` models (they carry `metricF1` as a custom object, which the scripts pass).
- The Keras and Torch trainers were meant to match; the compare scripts only test the explanation, not the training.
- `--no-plots` is accepted but does nothing in the analyze scripts, which make no plots (the options come from the shared parser).

## Still to decide

- Commit structure (nothing is committed): the env changes, the scripts, the README, and the `XAITorch` fix (a bug fix in the library, so possibly its own commit).
- When to remove `notebooks_TomsQoI/` (the README says: once the scripts are up and running).
- Whether to add the findings above to innvestigate issue #327.
- The README's status note ("none has been run end to end") must be updated as scripts are tested.
