# Handoff: XAIRT, the notebook-to-script conversion and the LRP research

Last updated at the end of the session of **2026-10-31**. Earlier session: 2026-09-20 (commit `31a3fc9`, its notes are condensed in section 12).
Read the README for what each script does; read this file for the state of things, what was found, what is running, and what is open.

---

## 0. Standing instruction for every session (from the owner): be super detailed

The owner asked that **every session write its notes in this file in full detail**, so that the next session never has to rediscover anything.
Concretely, before a session ends (and whenever something important happens), record here:

- **What was done and why**, with file paths and line numbers of code changes, and the reasoning behind each choice.
- **What was run**: the exact command or job script entry, the SLURM job id, the partition, where the output is, and the outcome with the actual numbers
  (not "it worked", but e.g. "agree to 2.4e-07 relative, Pearson 1").
- **What was *not* verified**: say plainly what only compiled, what only exited 0 without the numbers being checked, and what was never run.
- **What is running or queued** when the session ends: job ids, what each does, what it waits for, where its output will be, what a good and a bad
  outcome look like, and what to do if it fails.
- **Open questions and decisions** that are the owner's to make, and the owner's stated preferences (e.g. which queue to use).
- Keep old information that is still true; mark outdated information as outdated instead of silently deleting it.

Also keep this instruction at the top of the file.

**Back up the session transcript too (also from the owner).** The session of 2026-09-20 was lost: Claude Code deleted its transcript after the
default 30 days (`cleanupPeriodDays`). Since 2026-10-31, `~/.claude/settings.json` sets `"cleanupPeriodDays": 3650`, and copies are kept in
`/work/07665/shrey911/ls6/claude_session_backups/` (the `.jsonl` transcripts and the `memory/` notes). **Every session should refresh that copy
before it ends** (and after big milestones), with:

```bash
mkdir -p /work/07665/shrey911/ls6/claude_session_backups
cp -a ~/.claude/projects/-home1-07665-shrey911-XAIRT/*.jsonl ~/.claude/projects/-home1-07665-shrey911-XAIRT/memory \
      /work/07665/shrey911/ls6/claude_session_backups/
```

Sessions so far: `3cf0f372-6954-4299-afcf-169612297e29` (2026-10-31). Add each new session id here. A session can be resumed with
`claude --resume` (pick from a list) or `claude --continue` (the latest) from `/home1/07665/shrey911/XAIRT`, or from the VS Code extension's past
conversations. The transcript is a convenience; this file and git (commit and push) remain the durable record.

---

## 1. Where things stand, in short

- The environment works on the GPU nodes for both TensorFlow/Keras and PyTorch (section 2).
- All 10 fake-data scripts run (job 3544475). The question "do innvestigate (Keras) and Captum (PyTorch) give the same LRP?" is **answered**: they
  differ only in how the first layer's A1B0 treats negative inputs; with the same rule they agree to float precision (section 4.1).
- Captum (through `XAITorch`) can now run the owner's actual LRP method, LRP-A1B0-B (Bounded rule in the first layer), verified against innvestigate
  and a numpy formula (section 4.2).
- On the real ECCO data with the notebook's saved models: the default bounds (-1, 1) of the Bounded rule are violated by about 10% of the inputs and
  change the maps somewhat (section 4.3); and **many negative-class samples get exactly zero relevance** (section 4.4), now an open research question
  that the owner has long wanted to investigate (section 5).
- **Ten jobs are queued** (section 3): the first ECCO runs (OI, Keras-vs-Torch comparison, negative-class investigation on the saved models), and full
  retraining with 500 epochs in Keras and in Torch, followed by LRP comparisons and OI on the new networks, for all three cases (Keras, Torch,
  Keras network explained by Torch).
- **Nothing of the 2026-10-31 session is committed to git.** `git status` shows 19 modified files and 4 new ones (section 7). The owner has not yet
  asked for a commit.

---

## 2. How to run anything on this cluster (Lonestar6, TACC)

### 2.1 The login node: never run Python there

The login node is starved of resources. During the session it could not even fork (`echo hi` failed with exit 1, `bash: fork: retry: Resource
temporarily unavailable`), and a small Python command failed with `OpenBLAS error: Memory allocation still failed`. It also caps the virtual memory
of a process at 8 GB (`ulimit -v`), which breaks mamba. **Use it only for editing, `git`, `sbatch`/`squeue`/`scontrol`, `ls`, `cp`, and
`python -m py_compile` (parsing only, which worked).** Everything else goes to compute nodes through SLURM.

### 2.2 Which queue: owner's rule

- `gpu-a100-dev`: **only for small tests** (fake-data scripts, quick checks of new code). Short time limit, usually starts within minutes.
- `gpu-a100`: **all real runs**, i.e. any ECCO script on the real data, even a short one, and even if the dev queue looks emptier. The owner corrected
  the session once for putting ECCO runs on dev because it was emptier.
- Account: `Transfer-of-ECCO-glo` (set in the job scripts).
- A pending job can be moved without cancelling it: `scontrol update JobId=<id> Partition=gpu-a100`. `scontrol hold <id>` / `scontrol release <id>`
  stop and allow a pending job from starting. `scontrol update JobId=<id> Dependency=afterok:<other>` adds a dependency to a pending job.

### 2.3 Using the XAIRT env in a job (important, two traps)

The env is `/work/07665/shrey911/ls6/mambaforge/envs/XAIRT`. In a job script, do exactly this (it is in all three job scripts):

```bash
ENV=/work/07665/shrey911/ls6/mambaforge/envs/XAIRT
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH
```

- Trap 1: `conda activate XAIRT` in a batch job did **not** put the env's python first on `PATH` (job 3544455: every script failed with
  `ModuleNotFoundError: No module named 'numpy'`, since the base python was used).
- Trap 2: with `conda activate` or `mamba run`, TensorFlow prints `Could not load dynamic library 'libcublas.so.11'` and **sees no GPU** (it then
  silently runs on the CPU). The login profile loads the `cuda/11.4` module, whose `/opt/apps/cuda/11.4/lib64` is on `LD_LIBRARY_PATH` before the
  env's CUDA 11.8 libraries. Putting `$ENV/lib` first fixes it. Found with a diagnostic job (3544470) that tried four ways of entering the env; only
  the `LD_LIBRARY_PATH` one gave `GPUs: [GPU:0, GPU:1, GPU:2]`. PyTorch is not affected (its pip wheels bring their own CUDA 12.8 libraries).
- Verified in job 3544475: TensorFlow sees the 3 A100s, `torch.cuda.is_available()` is True (torch 2.8.0+cu128, CUDA 12.8), `import tensorflow`
  then `import torch` works (the NCCL clash of the previous session stays fixed), `import XAIRT` works. Versions: numpy 1.26.4, protobuf 3.20.3,
  keras 2.9.0, tensorflow 2.9.1, torch 2.8.0, captum 0.9.0, innvestigate 2.1.2. The installed TensorFlow is the conda-forge build
  `cuda112py310he87a039_0` (built for CUDA 11.2 / cuDNN 8), with cudatoolkit 11.8.0, cudnn 8.9.7.29, nccl 2.27.3.1 from conda-forge.
- GPU nodes: 3x NVIDIA A100-PCIE-40GB, driver 570.195.03.

### 2.4 TF32 on the A100s

TensorFlow 2.9 multiplies matrices in TF32 (10-bit mantissa) by default on Ampere GPUs. Seen in job 3544505: the largest difference between the
Keras and PyTorch probabilities of the *same* network went from 3.6e-07 (CPU, earlier) to **1.3e-03** once TensorFlow ran on the GPU. The LRP
relevances still agreed to about 2e-07 (innvestigate's graph evidently does not lose the precision in the same way), so the LRP conclusions stand.
TF32 is now turned off with `tf.config.experimental.enable_tensor_float_32_execution(False)` in:
`scripts_TomsQoI/fake_data_LRP_A1B0_compare.py`, `scripts_TomsQoI/eccov4r5_LRP_A1B0_compare.py`, `scripts_TomsQoI/eccov4r5_OI_keras.py`
(for the OI because the small differences would add up over thousands of gradient steps and make the Keras and Torch OIs differ).
**Not** turned off in the training scripts (`eccov4r5_LRP_A1B0_keras.py`, `simpleTests_*_keras.py`), so the networks trained there are trained
with TF32. PyTorch's default for matrix multiplication is full float32. **Open: decide whether training should also be float32** (section 11).

### 2.5 The job scripts (all submitted from the repository root, `/home1/07665/shrey911/XAIRT`)

The `#SBATCH -o/-e` paths are relative (`scripts_TomsQoI/results/logs/...`), so **always `cd` to the repository root before `sbatch`**, or the logs
go nowhere and the job fails. SLURM does not create the log directory, which is why `scripts_TomsQoI/results/logs/.gitkeep` is kept in git.

- `scripts_TomsQoI/run_easy_tests.slurm` (default partition `gpu-a100`, 45 min; pass `-p gpu-a100-dev` for tests): checks the env (versions, TF
  GPUs, torch CUDA, `import XAIRT`), then runs the fake-data scripts with small settings. With no arguments it runs all of them; with names it runs
  only those. Names: `fake_pos`, `fake_signed` (`fake_data_LRP_A1B0_compare.py` with and without `--positive-inputs`, 20 epochs, 1500 samples,
  50 explained), `LRP_keras`, `LRP_torch` (`simpleTests_LRP_*`, 2 epochs, 500 samples), `MWE_keras`, `MWE_torch`, `underst_keras`,
  `underst_torch`, `OI_keras`, `OI_torch` (`simpleTests_OI_*`, 1 epoch, 20000 samples). Note `LRP_keras` here means the simpleTests script, not the
  ECCO one. Output: `scripts_TomsQoI/results/easy_<jobid>/<name>.log` and the files each script writes there.
  Example: `sbatch -p gpu-a100-dev -t 00:20:00 scripts_TomsQoI/run_easy_tests.slurm fake_pos fake_signed`
- `scripts_TomsQoI/run_eccov4r5.slurm` (default `gpu-a100`, 2 h; pass a longer `-t` for training): runs the ECCO scripts that are named (it refuses
  to run with no names). Names and what they run:
  - `OI_keras`, `OI_torch`: `eccov4r5_OI_{keras,torch}.py` on the saved notebook models (`--source saved` / `saved-keras`), **small settings**:
    `--lags 0 --kind both --oi-iters 1000 --print-freq 100`.
  - `LRP_compare`: `eccov4r5_LRP_A1B0_compare.py --source saved --lags 0` on the saved notebook models.
  - `negative_class`: `eccov4r5_negative_class_LRP.py --source saved --n-lrp 300`, all 9 lags, on the saved notebook models.
  - `LRP_keras`, `LRP_torch`: `eccov4r5_LRP_A1B0_{keras,torch}.py` with **default settings = the notebook's**: all 9 lags, train 500 epochs,
    explain. Saved networks go to `<run dir>/LRP_keras/models/model{lag}_noL1.h5` and `<run dir>/LRP_torch/models/model{lag}_noL1.pt`.
  - `LRP_compare_trained` (needs `KERAS_MODELS`): the compare script on the networks of an `LRP_keras` run, all 9 lags.
  - `OI_keras_trained`, `OI_keras_in_torch_trained` (need `KERAS_MODELS`), `OI_torch_trained` (needs `TORCH_MODELS`): the OI scripts with the
    **notebook's settings** (all 9 lags, both kinds, `--eta 0.9999 --oi-iters 10000`) on the networks of an `LRP_keras` / `LRP_torch` run.
  - `KERAS_MODELS` / `TORCH_MODELS` are passed with `--export=ALL,KERAS_MODELS=$PWD/scripts_TomsQoI/results/ecco_<id>/LRP_keras/models`. A run
    that needs one and does not get it prints a message and is skipped (the `needs` function in the script).
  - Output: `scripts_TomsQoI/results/ecco_<jobid>/<name>.log` and `scripts_TomsQoI/results/ecco_<jobid>/<name>/` (the script's `--out-dir`).
- `env/create_env.slurm` (`gpu-a100`, 20 min): builds the env if it does not exist (`mamba env create -f env/XAIRT.yaml`, about 7 min), otherwise
  only checks it (versions, TF GPUs, torch CUDA, `import XAIRT`). Now uses the `PATH`/`LD_LIBRARY_PATH` way of section 2.3 instead of `mamba run`.
- Each job's `.out` in `scripts_TomsQoI/results/logs/` ends with one line per run: `<name>: exit <code> (<seconds> s)` and `done`.

### 2.6 Where output goes

- **Every script's default `--out-dir` is now `scripts_TomsQoI/results/<script name>/`** (e.g. `results/eccov4r5_OI_keras/`), wherever it is run
  from. Implemented by `default_out_dir()` in `scripts_TomsQoI/eccov4r5_common.py:58-62` (used by the ECCO, viz and data reader scripts) and the
  same expression in `simpleTests_common.py`, `fake_data_LRP_A1B0_compare.py`, `results_viz_qoi_pred.py`. Before, outputs went to folders
  relative to the current directory (`LRP_output_keras/`, `Results/`, ...). `--out-dir` still overrides it; the job scripts pass a name relative to
  their run directory.
- `scripts_TomsQoI/results/` is gitignored (except `results/logs/.gitkeep`), see `.gitignore`.
- Already in it: `easy_3544455`, `easy_3544459`, `easy_3544475`, `easy_3544480`, `easy_3544505`, `easy_3544532` (the fake-data runs of this session),
  `logs/` (all SLURM logs of this session, including the env diagnostic `XAIRT_tfgpu.3544470.out` and the old env builds `create_XAIRT_env.3458672/3458758`),
  `checks/` (the one-off ECCO rules check: `check_lrp_rules.py` and its outputs `check_lrp_rules.3544489.out` (part 1 only, crashed after) and
  `check_lrp_rules.3544506.out` (complete)).
- The notebook figures are saved by the scripts with the notebooks' colormaps, ranges and titles:
  - `eccov4r5_OI_{keras,torch}.py`: `OI_pos_<backend>.png`, `OI_neg_<backend>.png` (one map per lag, `RdBu_r`, -1..1), plus
    `OI_v4r5_{pos,neg}_newAnomalies_shuffleVal_reweight_avgInit_savedNN_<backend>.nc`.
  - `eccov4r5_LRP_A1B0_{keras,torch}.py`: `composite_{pos,neg}_<backend>.png` (cmocean balance, -1..1), `a1b0_b_{pos,neg}_<backend>.png` and
    `a1b0_{pos,neg}_<backend>.png` (mean normalized relevance, `jet`, 0..0.3), `metrics_<backend>.json` (accuracy, F1 per lag),
    `LRP_A1B0_newAnomalies_shuffleVal_reweight_<backend>.nc`.
  - The plotting code (`plot_maps` in `eccov4r5_common.py`) has **not run yet**; the queued jobs are its first test. matplotlib uses the `Agg` backend.

---

## 3. Jobs queued at the end of the 2026-10-31 session

State when the session ended: all pending (none had started). Check with `squeue -u $USER` and `sacct -j <id> -X -o JobID,State,ExitCode,Elapsed`.

| job | partition, limit | run (`run_*.slurm` name) | networks | waits for |
|---|---|---|---|---|
| 3544554 | gpu-a100, 2 h | `OI_keras OI_torch`: ECCO OI, lag 0, both kinds, 1000 steps | saved notebook models (Torch: copied with `keras_to_torch`) | - |
| 3544555 | gpu-a100, 2 h | `LRP_compare`: Keras vs Torch on the same network, LRP-A1B0-B and LRP-A1B0, lag 0 | saved notebook models | - |
| 3544556 | gpu-a100, 2 h | `negative_class`: the all-zero relevance investigation, all 9 lags | saved notebook models | - |
| 3544567 | gpu-a100-dev, 20 min | `run_easy_tests.slurm fake_pos LRP_torch`: test of the `Trainer.py` refactor (section 7) | fake data | - |
| 3544560 | gpu-a100, 12 h | `LRP_keras`: train 500 epochs and explain, all 9 lags | trained in this job (TF32 on) | - |
| 3544561 | gpu-a100, 12 h | `LRP_torch`: the same with Torch and Captum (innvestigate's rules) | trained in this job | 3544567 `afterok` |
| 3544562 | gpu-a100, 6 h | `LRP_compare_trained`: the networks of 3544560 explained by innvestigate and Captum, all 9 lags | from 3544560 | 3544560 `afterok` |
| 3544568 | gpu-a100, 6 h | `OI_keras_trained`: OI, notebook settings | from 3544560 | 3544560 `afterok` |
| 3544569 | gpu-a100, 6 h | `OI_keras_in_torch_trained`: the same, Keras networks copied into Torch | from 3544560 | 3544560 `afterok` |
| 3544570 | gpu-a100, 6 h | `OI_torch_trained`: the same, Torch (`--source saved-torch`) | from 3544561 | 3544561 `afterok` |

The three cases the owner asked for, for LRP and for OI: **Keras** (3544560 LRP, 3544568 OI), **Torch** (3544561 LRP, 3544570 OI), **Keras network
with Torch XAI** (3544562 LRP, 3544569 OI).

Why 3544561 waits for 3544567: 3544561 imports the refactored `src/XAIRT/Trainer.py` (`load_saved_model`), and 3544567 is the quick test that
training and reloading a Torch network still works. It was first put on hold and then given the dependency, so that it starts by itself if the test
passes, without anyone having to release it.

**If a job whose dependency failed stays pending with reason `DependencyNeverSatisfied`**: read the failed job's `.out`/`.err` in
`scripts_TomsQoI/results/logs/` and the per-run `.log` in its run directory, fix, `scancel` the stuck job, and resubmit (with the dependency or the
models directory of the new run).

What to check, in order, and what to expect:

1. **3544567** (`results/logs/XAIRT_easy.3544567.out`): both runs `exit 0`. If `LRP_torch` fails in `loadBestModel`/`load_saved_model`, the refactor
   is wrong (section 7, `Trainer.py`), and 3544561 and 3544570 will never start.
2. **3544556, negative class** (`results/ecco_3544556/negative_class.log` and `.../negative_class/negative_class.json`): per lag and class,
   `no_positive_contribution_share` (all explained samples, from a forward pass) should equal `all_zero_share_a1b0_b` and `all_zero_share_a1b0`
   (on the first 300 samples; compare with `no_positive_contribution_share_lrp_samples`, the same subset) if the mechanism of section 4.4 is right.
   `all_zero_share_a1b0_b_difference` tells whether explaining the logit difference removes the zeros. The first lines list the share of NaN wet
   points in each map of the published NetCDF file. This is the **first run** of the script; it may have first-run errors.
3. **3544555, Keras vs Torch on ECCO** (`results/ecco_3544555/LRP_compare.log`, `.../LRP_compare/agreement_compare.json`): expected
   `max_abs_prediction_diff` about 1e-6 or smaller (TF32 is off), and Pearson about 1 for both LRP-A1B0-B and LRP-A1B0 (Torch uses `BoundedRule`
   and `sign_split`). Anything clearly below 1 means a difference that the fake data did not show (e.g. something about the real network).
   First run of the modified script (`--source saved`).
4. **3544554, OI** (`results/ecco_3544554/OI_keras.log`, `OI_torch.log` and the PNG/NetCDF files): both runs exit 0; the predictions printed every
   100 steps should approach the desired label; the Keras and Torch OI maps should be close. Possible reason for a difference: Keras'
   `BinaryCrossentropy` clips probabilities to [1e-7, 1-1e-7], which zeroes the gradient once the output saturates; Torch's `BCELoss` does not clip.
   First run of the ECCO OI scripts and of `plot_maps`.
5. **3544560 / 3544561** (training): `metrics_keras.json`, `metrics_torch.json` (accuracy and F1 per lag; compare with each other and with the
   notebook), the LRP figures and NetCDF files. First full runs of the training scripts. No timing estimate exists: 500 epochs x 9 lags plus LRP of
   about 10000 samples per lag.
6. **3544562, 3544568-70**: as 3 and 4, on the new networks.

Not yet written: **a script that compares the two separately trained runs** (the NetCDF maps of 3544560 vs 3544561, and their accuracy and F1).
The compare scripts only compare explanations of one network.

---

## 4. Findings of the 2026-10-31 session

### 4.1 innvestigate vs Captum: resolved

The question from the previous session: do innvestigate (Keras) and Captum (PyTorch) give the same LRP-A1B0? Answered with
`scripts_TomsQoI/fake_data_LRP_A1B0_compare.py` (jobs 3544480 and 3544505; one seed, 20 epochs, 1500 samples, 40 features of which 4 informative,
50 explained samples, network 40-16-8-2 with ReLUs).

On the same Keras network (copied into PyTorch with `keras_to_torch`), with signed (standard normal) inputs, relative max difference and mean
Pearson correlation of the raw relevance:

| pair | LRP-A1B0 | LRP-A1B0-IB |
|---|---|---|
| innvestigate vs numpy formula with `(x_i w_ij)+` | 4.7e-07, r = 1 | 1.6e-07, r = 1 |
| Captum default vs numpy formula with `x_i (w_ij)+` | 1.8e-05, r = 1 | 6.0e-06, r = 1 |
| Captum with `sign_split` vs numpy `(x_i w_ij)+` | 9.7e-07, r = 1 | 1.4e-07, r = 1 |
| **innvestigate vs Captum with `sign_split`** | **5.0e-07, r = 1** | **1.8e-07, r = 1** |
| innvestigate vs Captum default | 1.0, r = 0.43 (min -0.29) | 1.0, r = 0.36 (min -0.49) |
| Keras network vs separately trained Torch network | 1.0, r = 0.18 | 1.0, r = 0.17 |

With `--positive-inputs` (X uniform in [0, 1]) all of innvestigate, Captum default, Captum `sign_split` and both numpy formulas agree to about 2e-07.

Conclusions:
- The only difference between the libraries is the positive contribution of A1B0: innvestigate uses `(x_i w_ij)+` (positive weights on positive
  inputs plus negative weights on negative inputs), which is the definition of Bach et al. 2015 and Montavon et al. 2019; Captum's
  `Alpha1_Beta0_Rule` uses `x_i (w_ij)+` (weights clamped to positive, applied to the input as it is), which is the same only for inputs >= 0. All
  hidden layers have inputs >= 0 (after a ReLU), so only the **first layer** differs, and only for signed inputs such as SST anomalies.
- Both libraries put the raw bias in the denominator and let it take a share of the relevance (innvestigate's positive/negative layer copies split
  the bias into b+ and b-, which add back to b); both pass relevance through ReLUs unchanged; the explained neuron (max output) is the same.
- Captum's rule on signed inputs is unstable: largest relevance 106 against about 1 for innvestigate, because the denominator `sum_i x_i (w_ij)+` can
  be close to zero or negative. Its agreement with its own numpy formula is also slightly worse (1.8e-05) for the same reason (round-off amplified).
- In the owner's terms: for negative inputs innvestigate follows the published rule; Captum is not "buggy" but applied outside the inputs its rule is
  meant for (Captum documents it for post-ReLU layers).
- The previous session's `XAITorch` bug fix (rules re-attached before every sample, because Captum deletes them after every `attribute()` call) is
  confirmed by these runs: Captum matches the numpy formula on all 50 samples, which is impossible if only the first sample got the rule. The
  regression test asked for in the previous handoff is therefore covered.

### 4.2 The Torch `BoundedRule` is correct

`XAI.py` now has a Captum version of innvestigate's Bounded (z^B) rule (section 7). Verified in job 3544532 with `fake_data_LRP_A1B0_compare.py`:
LRP-A1B0-B (Bounded first layer, A1B0 ignoring the bias in the other layers) by innvestigate, by `XAITorch` with the new rule on the same network,
and by the numpy formula, relative max difference, Pearson 1 everywhere:

| inputs | bounds | innvestigate vs numpy | Captum vs numpy | innvestigate vs Captum |
|---|---|---|---|---|
| positive | (-1, 1) | 1.8e-07 | 1.8e-07 | 2.6e-07 |
| positive | training range (1.3e-05, 0.99999) | 1.6e-07 | 1.6e-07 | 1.9e-07 |
| signed | (-1, 1) | 1.4e-07 | 2.5e-07 | 2.6e-07 |
| signed | training range (-4.39, 4.15) | 1.6e-07 | 1.6e-07 | 1.7e-07 |

The signed inputs with (-1, 1) bounds are often outside the bounds, so the implementation matches innvestigate also where the rule's premise fails.
**Not tested: per-input bounds** (one low/high per grid point; the code supports arrays). Not yet run on the ECCO data (job 3544555 is the first).

### 4.3 The LRP rules of the ECCO scripts, checked on the real data and the saved models

One-off check, `scripts_TomsQoI/results/checks/check_lrp_rules.py` (jobs 3544489, part 1 only, and 3544506, complete; output
`results/checks/check_lrp_rules.3544506.out`). Data: X is (11263 samples, 31811 wet points) of SST anomalies in degC, not scaled.

- **Inputs vs the bounds.** innvestigate's `input_layer_rule='Bounded'` means `BoundedRule(low=-1, high=1)` (read in
  `innvestigate/analyzer/relevance_based/relevance_rule.py`, class `BoundedRule`, and `relevance_analyzer.py` where `"Bounded"` maps to it). The
  z^B rule assumes every input is within [low, high] (that is what makes each term `w(x - l)` or `w(x - h)` non-negative). Measured:
  - 10.4% of all values are outside [-1, 1] (so the draft's "90% of the data is in [-1, 1]" is right);
  - **every sample** has at least one value outside;
  - 93% of the grid points leave [-1, 1] at some time; the median over grid points of max |anomaly| is 2.28 degC;
  - training min/max: -13.83 / 8.17 degC; percentiles 0.1/1/5/50/95/99/99.9%: -3.11, -1.78, -1.01, -0.001, 1.02, 1.89, 3.54.
- **Effect of the bounds** (lags 0 and 60, first 400 explained samples per class, LRP-A1B0-IB in the hidden layers): Bounded (-1, 1) gives 0.5-0.7%
  negative relevance (share of the relevance mass), Bounded with the training range gives none. Correlation of the mean normalized relevance maps
  (the ones that are plotted) between Bounded (-1, 1) and Bounded (training range): **0.898 (lag 0 pos), 0.867 (lag 60 pos), 0.781 (lag 60 neg)**.
  Bounded (-1, 1) vs plain A1B0-IB / A1B0 (no input rule): 0.916 (lag 0 pos), 0.879 (lag 60 pos), 0.896 (lag 60 neg); A1B0 and A1B0-IB gave the
  same correlations to 3 digits (the biases are small, probably because of the L2 penalties). So **the choice of bounds does change the maps**
  noticeably, which contradicts the draft's sentence that it does not (section 6).
- **The explained logit is never negative**: share of explained samples whose predicted-class logit (before the softmax) is < 0 is 0.000 at lags
  -60, 0, 60, 180, both classes; median logit 1.73-1.92, median logit difference 3.46-3.83. (Raised as a possible issue because A1B0 starts from that
  logit, and a negative start would flip the sign of all relevance.) **Dismissed.**

### 4.4 All-zero relevance in the negative class

From the same check (lags 0 and 60, first 400 explained samples per class, the counts that `XAIKeras` prints as "Number of all-zero samples"):
- lag 0, neg: **400 of 400 (100%)** samples have all-zero relevance, for every method (Bounded (-1, 1), Bounded (data range), A1B0-IB, A1B0);
- lag 60, neg: **113 of 400 (28%)**, for every method;
- positive class: 0 at both lags.

Likely mechanism (not yet confirmed, job 3544556 does that): A1B0 distributes only the positive contributions `(h_i w_ic)+` of the last hidden
layer into the logit of the explained class c. If no active hidden neuron has a positive weight into the negative logit, there is nothing positive
to distribute: the negative logit is positive only through the output bias, every input gets zero relevance (with the bias kept, all relevance goes
to the bias). This holds whatever rule the first layer uses, which is why every method gives the same count.

Consequence for the published results: `normalized_mean_relevance` (`scripts_TomsQoI/eccov4r5_common.py:293`, the notebook's `LRP_normalize`)
divides each sample by its max (0/0 = NaN for an all-zero sample) and takes `nanmean`, so all-zero samples are silently dropped. **The published
negative-class maps therefore average over only the samples that LRP could explain, or are entirely NaN where none could** (e.g. lag 0, if the
first 400 are representative). In the check, the lag 0 neg correlations came out NaN for that reason.

---

## 5. Open research question: why does LRP give no relevance to the negative class?

**The owner has long wanted to investigate this; it is a goal of the project, not a side note.**

Plan:
1. **Measure** (job 3544556 does it on the saved notebook models): for all explained samples and all 9 lags, per class, the share with no positive
   contribution into the explained logit (forward pass); the all-zero share of LRP-A1B0-B and LRP-A1B0 on a subsample (300 per class) to confirm;
   and the share of NaN wet points in each map of the published file
   `examples_TomsQoI/LRP_output_forHelen/LRP_A1B0_IB_B_newAnomalies_shuffleVal_reweight.nc`.
2. **Confirm the mechanism**: for those samples, the hidden activations, the weights into the negative logit, the bias share of the logit
   (job 3544556 reports `bias`, `bias_share_of_logit_median`, `active_neurons_median` per hidden layer). Then ask *why the network learned this*:
   is "negative = nothing fires" a result of the class weights of training (`class_weights` in `eccov4r5_common.py`), of the L2 penalty of 10 on
   the first layer's weights and biases (`make_layers`), of the QoI definition (negative = the 61-day running mean of the anomaly at tile 10,
   j 1, i 43 is < 0)?
3. **Remedies to compare**:
   - explain the **logit difference**: replace the 2-unit last layer by one unit with weights `w_c - w_other` and bias `b_c - b_other` (the same
     predictions; neurons that vote against the other class become evidence for c). Job 3544556 already reports its all-zero share and the
     correlation of its positive-class mean map with the current one (`difference_model` in `eccov4r5_negative_class_LRP.py`);
   - keep the bias as a relevance sink and report its share;
   - a network with a single sigmoid output;
   - judge them against the adjoint sensitivities and the OI of the negative class.
4. Decide whether the published negative-class maps and their interpretation in the draft need to change.

The new script for this is `scripts_TomsQoI/eccov4r5_negative_class_LRP.py` (section 7). It has not run yet.

---

## 6. The draft's LRP appendix (the owner shared the LaTeX of "LRP Rule Selection")

The appendix says: A1B0 (Bach et al. 2015) in all layers but the first, bias ignored in numerator and denominator "due to a bug in innvestigate";
the Bounded rule in the first layer "since this yields better results"; SST anomalies have zero mean and finite variance, hence are bounded; tested
various l_i, h_i, "they do not affect the results much", so the defaults l = -1, h = 1 are kept. A commented-out todo discusses whether Toms et
al.'s claim that Bounded needs zero mean and unit variance is right (it is not in Montavon et al. 2017, which only needs box bounds).

Suggestions made in the session (none applied to the draft, it is the owner's):
1. **Do not call the bias effect an innvestigate bug.** Captum does the same (section 4.1): the raw bias is in the denominator and takes a share of
   the relevance, the usual convention of treating the bias as a neuron (Bach et al. 2015). Suggested wording: "When biases are included, part of
   the relevance is assigned to them, so the relevance of the inputs does not sum to the network output. To keep relevance conserved across layers we
   ignore the biases in both the numerator and the denominator."
2. The A1B0 equation `a_i w_ij+` is right for the hidden layers because their inputs are >= 0, where it equals `(a_i w_ij)+`; say so.
   Notation: the denominators sum over i while i is the free index, use i'; in the Bounded equation the numerator has `x_i w_ij` and the
   denominator `z_ij`, use one; "pixel values" should be "grid-point values".
3. "Zero mean and finite variance, hence bounded" does not follow; a finite training set always has box bounds (its min and max), which is all
   Montavon et al. 2017 require (the todo is right that they never ask for zero mean or unit variance). But the rule needs every input inside
   [l_i, h_i], and with (-1, 1) about 10% are not (section 4.3). Either use the data bounds (global or per grid point), or support "does not affect
   the results much" with numbers; the numbers so far (map correlations 0.78-0.90) say it does, somewhat.
4. "Yields better results": say what better means (agreement with the adjoint?).
5. Possibly a sentence on the negative class (section 5) once understood.

---

## 7. Code changes of the 2026-10-31 session, file by file (all uncommitted)

### Library, `src/XAIRT/`
- `utils.py:43-64`: `Keras_GradientDescent_useGradientTape` was `@tf.function` on top of `@beartype`, which failed when tracing with
  `TypeError: Tensor.__tf_tracing_type__() missing 1 required positional argument: 'signature_context'`: beartype keeps the annotated types (the
  `Tensor` class) as hidden default arguments of its wrapper, and tf.function tried to trace them. Now a plain `@tf.function`
  `_keras_gradient_of_loss` (line 47) and the beartype-checked `Keras_GradientDescent_useGradientTape` (line 59) that calls it. Same interface.
  Fixed `simpleTests_OI_keras.py` (verified, job 3544475) and is used by `eccov4r5_OI_keras.py`.
- `XAI.py:209` `Alpha1_Beta0_SignSplit_Rule(Alpha1_Beta0_Rule)`: innvestigate's A1B0 as a Captum rule. In `_manipulate_weights` it sets each input
  column of the weight to `w+` where that input is > 0 and `w-` where it is < 0 (`torch.where(x[0] > 0, w.clamp(min=0), w.clamp(max=0))`), so it
  needs a batch of one sample (raises otherwise), which is how `XAITorch` calls Captum. Verified (section 4.1).
- `XAI.py:229` `BoundedRule(PropagationRule)`: innvestigate's z^B rule as a Captum rule, `R_i = sum_j (x_i w_ij - l_i w_ij+ - h_i w_ij-) /
  sum_i' (...) R_j`, bias ignored. How it fits Captum's hooks (Captum computes the relevance of an input as input x gradient): `_manipulate_weights`
  keeps the weights and replaces the bias by `-sum_i (l_i w_ij+ + h_i w_ij-)`, so the layer's modified output is the z^B denominator; the output hook
  stores `s_j = R_j / z_j`; the input hook returns `x_i (W^T s)_i - l_i (W+^T s)_i - h_i (W-^T s)_i`. low/high are scalars or arrays of one value
  per input. Needs a layer with a bias (raises otherwise; Captum restores the original state dict after `attribute()`). Verified (section 4.2).
- `XAI.py:366` `XAITorch._attach_lrp_rules`: `lrp.alpha_1_beta_0` and `lrp.alpha_1_beta_0_IB` take optParams `{'sign_split': True}`; all LRP methods
  take `{'input_layer_rule': 'Bounded'}` (low -1, high 1, as innvestigate) or `{'input_layer_rule': (low, high)}`, which puts `BoundedRule` on the
  **first Linear layer only** (as innvestigate). Other input layer rules raise `NotImplementedError`. Imports added at the top:
  `PropagationRule`, `torch.Tensor`, `typing.List, cast`. Docstring of `XAITorch` updated.
- `Trainer.py`: `TrainTorchFullyConnectedNN._build_model_modules` is now a `@staticmethod` taking `layers_cfg` (line 362); new
  `@staticmethod load_saved_model(path, layers_cfg)` (line 613) rebuilds the network and loads a `.pt` state dict (eval mode, CPU);
  `loadBestModel` (line 632) now calls it. Purpose: load the Torch networks saved by `eccov4r5_LRP_A1B0_torch.py` without a trainer
  (`--source saved-torch`). **Being tested by job 3544567.**

### Scripts, `scripts_TomsQoI/`
- `fake_data_LRP_A1B0_compare.py`: rewritten and merged with the session's `fake_data_LRP_same_rules.py` (deleted). Cases on one Keras network:
  `keras` (innvestigate), `keras_torch` (Captum default), `keras_torch_split` (Captum `sign_split`), `numpy_split`, `numpy_wplus` (formulas,
  `numpy_a1b0`), and `torch` (separately trained). All on raw relevance. Then LRP-A1B0-B with bounds (-1, 1) and the training range: `keras`,
  `keras_torch` (`BoundedRule`), `numpy_bounds`. TF32 off. Writes `agreement_fake.json`.
- `eccov4r5_LRP_A1B0_keras.py`: comment on `METHODS` (both methods also run in Torch now); `make_get_model` renames the layers of a loaded model to
  `saved_<name>` (line 69), because `innvestigate.model_wo_softmax` on a loaded model failed with `ValueError: The name "dense" is used 2 times in
  the model` (its new layer gets the automatic name 'dense' in a fresh session).
- `eccov4r5_LRP_A1B0_torch.py`: `METHODS` is now LRP-A1B0-B (`input_layer_rule: 'Bounded'`) and LRP-A1B0 (`sign_split: True`), as the Keras script;
  `make_get_model` (line 58) has `--source saved-torch` (loads `model{lag}_noL1.pt` with `load_saved_model`).
- `eccov4r5_LRP_A1B0_compare.py`: methods = the Keras script's; the Torch side adds `sign_split` (`explain_torch`); new `--source saved|train`
  (default `saved`, the networks of `--saved-models-dir`) instead of always training 500 epochs; uses `make_context` and `make_get_model`; TF32 off;
  docstring rewritten (agreement to float precision is now expected).
- `eccov4r5_OI_keras.py`: TF32 off. `eccov4r5_OI_torch.py` and `eccov4r5_analyze_torch.py`: sources `saved-keras`, `saved-torch`, `train`.
- `eccov4r5_common.py`: `RESULTS_DIR` and `default_out_dir()` (lines 58-62), default `--out-dir`; help text of `--source` mentions `saved-torch`.
- `eccov4r5_viz_common.py`, `eccov4r5_dataReader.py`, `results_viz_qoi_pred.py`, `simpleTests_common.py`: default `--out-dir` in `results/`.
- New `eccov4r5_negative_class_LRP.py` (section 5): `mechanism()` (forward pass: positive contributions into logit c, logit, bias, active neurons),
  `difference_model()` (a fresh copy of the network with the logit-difference output; fresh layers because innvestigate refuses layers used in two
  models), LRP on `--n-lrp` samples per class, published-file NaN shares relative to the wet points. Writes `negative_class.json`. Not run yet.
- New `run_easy_tests.slurm`, `run_eccov4r5.slurm` (section 2.5).

### Other
- `env/create_env.slurm`: env used through `PATH`/`LD_LIBRARY_PATH` (section 2.3) instead of `mamba run`; logs to `scripts_TomsQoI/results/logs/`.
- `env/diagnose_tf_gpu.slurm`: written and deleted in the session (its finding is in section 2.3; its log is `results/logs/XAIRT_tfgpu.3544470.out`).
- `.gitignore`: `scripts_TomsQoI/results/*` ignored, except `results/logs/.gitkeep`.
- `README.md`: fake-data entry rewritten (the libraries' one difference), ECCO entry (Torch uses innvestigate's rules), a paragraph on the results
  directory and the job scripts.
- Memory of the assistant (outside the repo): the owner's queue rule is saved.

---

## 8. Which scripts have run (status at the end of 2026-10-31)

| script | status |
|---|---|
| `fake_data_LRP_A1B0_compare.py` | runs, numbers checked (sections 4.1, 4.2; jobs 3544480, 3544505, 3544532) |
| `simpleTests_LRP_{keras,torch}.py`, `LRP_manual_MWE_{keras,torch}.py`, `understandingLRP_{keras,torch}.py`, `simpleTests_OI_{keras,torch}.py` | run, exit 0 on the GPU (job 3544475); **their numbers were not compared with the hand computation in this session** (the previous session checked `LRP_manual_MWE_*` on the login node, section 12) |
| `eccov4r5_OI_{keras,torch}.py` | queued (3544554 small; 3544568-70 full); never run |
| `eccov4r5_LRP_A1B0_compare.py` | queued (3544555, 3544562); never run |
| `eccov4r5_negative_class_LRP.py` | queued (3544556); never run |
| `eccov4r5_LRP_A1B0_{keras,torch}.py` | queued (3544560, 3544561); never run |
| `eccov4r5_analyze_{keras,torch}.py` | never run |
| `eccov4r5_dataReader.py` | never run (`SST_all.nc` exists in `/work/07665/shrey911/ls6/LRP_eccov4r5_data/`, so `--build-sst` is not needed) |
| `eccov4r5_results_viz.py`, `eccov4r5_results_viz_new.py`, `results_viz_qoi_pred.py` | never run |
| `scripts_TomsQoI/results/checks/check_lrp_rules.py` (one-off, not in git) | ran (job 3544506), section 4.3 |

---

## 9. Data and paths

- ECCO data: `/work/07665/shrey911/ls6/LRP_eccov4r5_data/` (`SST_all.nc` 4.8 GB, `GRID/`, the daily mds files) and
  `/work/07665/shrey911/ls6/LRP_eccov4r4_data/thetaSurfECCOv4r4.nc`. Loading and computing the anomalies takes about 2-3 minutes in a job.
- **Saved notebook models**: `examples_TomsQoI/LRP_output_forHelen/saved_models/model{lag}_noL1.h5` for lags -60, -30, 0, 30, 60, 90, 120, 150,
  180 (2 MB each, dated 2026-03-25); a second set directly in `examples_TomsQoI/` (lag 0 is byte-identical to the first set; the others were not
  compared). The lag -60, 0, 60 and 180 models loaded fine (job 3544506); the others will first load in job 3544556. The published maps:
  `examples_TomsQoI/LRP_output_forHelen/LRP_A1B0_IB_B_newAnomalies_shuffleVal_reweight.nc`.
- `examples_TomsQoI/` is untracked and the owner calls it stale, but it holds the only copies of these models (`*.h5` is gitignored). **Backed up on
  2026-10-31** to `/work/07665/shrey911/ls6/XAIRT_examples_TomsQoI_backup_2026-10-31/` (the whole folder, 29 files, 61 MB, byte-compared).
  `/work` is not purged but not backed up by TACC either; a copy off the cluster was suggested.
- The scripts' `--saved-models-dir` default is `LRP_output_forHelen/saved_models` (relative to the current directory, the notebooks' path), so pass
  it explicitly (the job scripts do).
- Not in this checkout: the OI result files and the older `LRP_A1B0*.nc` / `OI_*small*.nc` that `eccov4r5_results_viz.py` plots (its panels are
  skipped with a message). `qoi_pred.nc` for `results_viz_qoi_pred.py` is written by `eccov4r5_analyze_{keras,torch}.py` as
  `qoi_pred_{backend}.nc` (use `--file`).
- Home quota 10 GB (about 74% used at the start of the session); the results so far are about 1 MB.

---

## 10. Things likely to break, or to keep in mind

- First runs of every ECCO script (section 8): expect a first-run error or two.
- `eccov4r5_OI_*` with the notebook settings: 10000 steps x 9 lags x 2 kinds; the Keras one calls a `tf.function` per step. Time unknown (6 h limit).
- Training time of `eccov4r5_LRP_A1B0_*` unknown (12 h limit).
- `XAIKeras` and `XAITorch` explain one sample at a time (a loop in `analyze_samples`), so LRP of about 10000 samples per lag is slow.
- The Keras and Torch trainers were made to match (previous session), but nothing has compared their training yet (accuracy, F1, maps).
- `--no-plots` does nothing in the analyze scripts (they make no plots).
- `BoundedRule` replaces the layer's bias, so a first layer without a bias raises.
- `Alpha1_Beta0_SignSplit_Rule` needs a batch of one sample.

---

## 11. Decisions that are the owner's

- **The bounds of the Bounded rule**: keep (-1, 1), use the training range, or per grid point (section 4.3; the owner said "then we can discuss more
  principled ways" once Keras and Torch match on ECCO).
- **The negative class** (section 5): how to explain it, and whether the published maps change.
- **TF32 in training** (section 2.4).
- **Commit structure**: nothing of 2026-10-31 is committed. Possible split: library fixes (`utils.py` tracing fix), library features
  (`XAI.py` rules, `Trainer.py` loader), scripts, job scripts and results directory, env script, README/HANDOFF.
- From the previous session, still open: when to remove `notebooks_TomsQoI/` (the README says once the scripts run); whether to add the findings to
  innvestigate issue #327 (note: section 4.1 says the bias behaviour is a convention shared with Captum, not an innvestigate-only bug); updating the
  README status note as scripts get tested.

---

## 12. History: the session of 2026-09-20 (condensed; outdated items marked)

- **Conversion**: all 12 notebooks of `notebooks_TomsQoI/` got script counterparts in `scripts_TomsQoI/` (Keras and Torch versions where a backend is
  needed), written without an environment that could run them.
- **Environment**: `env/XAIRT.yaml` (conda-forge TensorFlow 2.9.1 `cuda112py310`, innvestigate 2.1.2, captum 0.9.0, torch pinned to 2.8.0). The first
  build (job 3458672) had `import torch` after `import tensorflow` fail with `undefined symbol: ncclCommResume`: the unpinned pip torch 2.14+cu130
  bundles NCCL 2.30, conda has NCCL 2.27.3, both are `libnccl.so.2`, and TensorFlow loads first. Fixed by `pip install torch==2.8.0` (needs exactly
  NCCL 2.27.3, CUDA 12.8 build; the A100 driver 570 supports 12.8, not 13). *Verified on a GPU node in 2026-10-31 (section 2.3).* The env cannot be
  built on the login node (8 GB virtual memory cap breaks mamba); the pip cache goes to `/work/07665/shrey911/ls6/pip_cache` (10 GB home quota).
  The old `py310_LRP` env is broken (numpy 2, protobuf 7, keras 3) and must not be used.
- **`XAITorch` bug** (fixed then, *confirmed in 2026-10-31, section 4.1*): the LRP rule was attached once, but Captum deletes the rules after every
  `attribute()` call, so only the first sample got A1B0 and the rest the default epsilon rule. Any Torch LRP-A1B0 result from before the fix is wrong.
- **Login-node runs** (with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_THREADS=1
  CUDA_VISIBLE_DEVICES=`, otherwise "Thread creation failed"; *now: use compute nodes*): `LRP_manual_MWE_keras.py --epochs 2 --n-samples 500`:
  innvestigate matches the hand computation (WSquare and Bounded (-1, 1) to 1e-7; `lrp.alpha_1_beta_0` = "one bias in the denominator, none in the
  numerator", with `R_last - sum(a)` equal to the bias share 0.144; `_IB` = "without bias", residual 0): innvestigate issue #327 reproduced.
  `LRP_manual_MWE_torch.py`: Captum's A1B0, Z, epsilon match the hand computation (but the predicted class had bias 0, NonNeg bias).
- **The hypothesis** (resolved in 2026-10-31, section 4.1): innvestigate splits the input by sign, Captum clamps weights; the first layer would differ
  for signed inputs.
- **What changed relative to the notebooks** (still true):
  - The notebooks call names that no longer exist (`TrainFullyConnectedNN`, `XAIR`, `GradientDescent_useGradientTape`, `TrainOI_useGradientTape`).
  - Anomalies: the viz notebooks used the old `anomalize` for the correlation maps; the LRP and OI results use the new ones, which are correct. The
    scripts use the new ones everywhere; the old function was removed.
  - simpleTests fake data uses a seeded generator, so numbers differ from the notebooks.
  - In the LRP fake data class 0 (`x1 + 2 x2 >= 3`) is empty, so the network always predicts class 1; kept, with a comment.
  - `pred_class` is the argmax everywhere (one notebook hard-coded 1). The Adam demo recomputes the gradient every step. The OI demo uses the trained
    network (the notebook cloned it, which does not copy weights).
  - `understandingLRP` gave innvestigate bounds (-0.4, 0.6) but computed (0.4, 0.6) by hand; both are now run. Inputs are in [0, 1], so neither
    pair contains them.
  - `eccov4r5_results_viz_new.ipynb` bugs fixed in the script: stale lag index in the composite figures; the OI negative figure was titled "OI-pos".
  - Not converted: raw SST plots and the old-vs-new anomaly comparison of the data reader notebook; the commented-out animation of the viz notebook.
  - `eccov4r5_common.py`: `load_data` = `load_anomalies` + `make_qoi`; `add_model_args`, `make_context`, `train_on_lag`; LRP scripts got `make_get_model`.
