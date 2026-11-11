# Handoff: XAIRT, the notebook-to-script conversion and the LRP research

Last updated at the end of the session of **2026-10-31**; continued on **2026-11-01** and **2026-11-10** (same session: the miniature test job, the follow-up checks and the results of all queued jobs, section 3). Earlier session: 2026-09-20 (commit `31a3fc9`, its notes are condensed in section 12).
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
- `scripts_TomsQoI/run_eccov4r5_mini.slurm` (added 2026-11-01; `gpu-a100-dev`, 1 h; no arguments): **quick test of every
  `run_eccov4r5.slurm` run before queuing the long ones.** Same scripts and options, small settings: `--lags 0 60`, `--epochs 3`,
  `--oi-iters 20 --print-freq 10`, `--n-lrp 20` (`LRP_compare` on the saved models: lag 0 only). Runs, in order: `OI_keras`, `OI_torch`,
  `LRP_compare`, `negative_class`, `LRP_keras`, `LRP_torch`, then `LRP_compare_trained`, `OI_keras_trained`, `OI_keras_in_torch_trained`,
  `OI_torch_trained` on the 3-epoch networks the job just trained (`$RUN_DIR/LRP_{keras,torch}/models`), which tests the KERAS_MODELS /
  TORCH_MODELS hand-off of the real jobs. A failing script does not stop the others; the job log prints `<name>: exit <code> (<s> s)` and, on
  failure, the last 5 lines of `<name>.log`. Output: `scripts_TomsQoI/results/mini_<jobid>/`, log `results/logs/XAIRT_mini.<jobid>.out`.
  Why: the dev queue allows only **one running job per user**, so all the miniatures are in one job; and the real `gpu-a100` jobs wait long
  (on 2026-11-01, 265 of 434 pending jobs had higher priority), so a first-run error found there costs a day. **When adding a run to
  `run_eccov4r5.slurm`, add its miniature here too.** The first run of it (job 3545828, 2026-11-01) was submitted from a scratch copy with
  the output in `/work/07665/shrey911/ls6/XAIRT_mini/` (`mini.3545828.out`, `run_3545828/`) instead of `results/`; see section 3.
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
| 3544567 | gpu-a100-dev, 20 min | `run_easy_tests.slurm fake_pos LRP_torch`: test of the `Trainer.py` refactor (section 7). **Done, passed** (both exit 0, numbers as before) | fake data | - |
| 3544560 | gpu-a100, 12 h | `LRP_keras`: train 500 epochs and explain, all 9 lags | trained in this job (TF32 on) | - |
| 3544561 | gpu-a100, 12 h | `LRP_torch`: the same with Torch and Captum (innvestigate's rules) | trained in this job | 3544567 `afterok` |
| 3544562 | gpu-a100, 6 h | `LRP_compare_trained`: the networks of 3544560 explained by innvestigate and Captum, all 9 lags | from 3544560 | 3544560 `afterok` |
| 3544568 | gpu-a100, 6 h | `OI_keras_trained`: OI, notebook settings | from 3544560 | 3544560 `afterok` |
| 3544569 | gpu-a100, 6 h | `OI_keras_in_torch_trained`: the same, Keras networks copied into Torch | from 3544560 | 3544560 `afterok` |
| 3544570 | gpu-a100, 6 h | `OI_torch_trained`: the same, Torch (`--source saved-torch`) | from 3544561 | 3544561 `afterok` |

**Added 2026-11-01:** none of the jobs above had started by 2026-11-01 (all `gpu-a100` nodes busy, 434 pending jobs, 265 with higher
priority than ours). So **job 3545828** (`gpu-a100-dev`, 1 h) runs the miniature of all of them (`run_eccov4r5_mini.slurm`, section 2.5), to find
first-run errors before the real jobs get a node. Log: `/work/07665/shrey911/ls6/XAIRT_mini/mini.3545828.out`; per-run logs and outputs:
`/work/07665/shrey911/ls6/XAIRT_mini/run_3545828/`. If a miniature fails, fix the script before its real job starts (the real jobs read the
scripts when they start, not when submitted).

**Result of 3545828 (ran 2026-11-01 on c301-003, about 20 min): 10 of 11 exit 0, `negative_class` exit 1.**

| run | exit, time | what it showed |
|---|---|---|
| `OI_keras` | 0, 308 s (mostly the first data load) | PNGs and NetCDFs written, maps look right (`plot_maps` works). Colorbars much taller than the maps (layout only). Also creates an empty `models/` in its out-dir (harmless) |
| `OI_torch` | 0, 33 s | Same start as Keras to 7 digits (iter 0: lag 0 pos 0.81718, lag 60 pos 0.70839, neg 0.98041 / 0.98175), **but the trajectories differ**: after 20 steps lag 60 pos Keras 0.8385 vs Torch 0.8567, lag 0 pos 0.8954 vs 0.8938, lag 0 neg 0.98053 vs 0.98045. Same update rule (`x - eta*grad`, BCE), so the gradients differ, not by rounding. **Explained (from the Keras 2.9 source, not yet confirmed numerically):** Keras' `softmax` stores its input logits on its output (`output._keras_logits = x`, `keras/activations.py:91`), and `backend.binary_crossentropy` uses them whenever present (`keras/backend.py:5272-5283`): `tf.nn.sigmoid_cross_entropy_with_logits(target, z)`. So **Keras' OI loss is the mean over the 2 outputs of BCE(sigmoid(z_k), t_k)**, for desired [1,0]: ½[softplus(-z0) + softplus(z1)], pushing z0 up and z1 down separately. **Torch's `BCELoss` works on the softmax probabilities**: ½[-log p0 - log(1-p1)] = -log sigmoid(z0 - z1), which depends only on the logit difference. Same forward pass (identical iter 0), different loss, different gradient. **The notebook's published OI used the Keras one.** Training is not affected: the ECCO scripts train with `categorical_crossentropy` (`eccov4r5_common.py:38`), where Keras' logits shortcut is correct and Torch uses `_softmax_crossentropy`. **Owner decision**: either make Torch's OI reproduce Keras (`BCEWithLogitsLoss` on the pre-softmax logits, i.e. `model_wo_softmax_torch`) or make both use the probabilities (Keras: `BinaryCrossentropy` on an output without `_keras_logits`, e.g. a Lambda softmax or `tf.identity` of the output). Job 3544554 (1000 steps) shows how much the final maps differ |
| `LRP_compare` (saved models, lag 0) | 0, 140 s | **innvestigate vs Captum on ECCO: identical.** max prob diff 2.09e-06; pos (n 4733) and neg (n 5770), LRP-A1B0-B and LRP-A1B0: pearson_mean, pearson_min, cosine_mean, pearson_mean_map all 1.0000 (pearson_min 0.9999999999995) |
| `negative_class` | **1**, 27 s | crashed in `difference_model`: `RuntimeError: Random ops require a seed to be set when determinism is enabled` (Keras' random initializer of the new Dense layers, graph mode + op determinism). **Fixed**: zero initializers (weights are set right after). Fix tested by 3545869, see below. Before the crash it showed the published file has no NaN at wet points (section 4.4 correction) |
| `LRP_keras` (3 epochs, lags 0 60) | 0, 105 s | models `model{0,60}_noL1.h5`, all 6 PNGs, metrics, NetCDF. Acc/F1: lag 0 0.733/0.761, lag 60 0.790/0.809. All-zero relevance over **all** explained samples: neg 34.8% (1662) at lag 0, 2.4% (122) at lag 60, both methods; pos 0 |
| `LRP_torch` (3 epochs) | 0, 208 s | models `.pt`, all 6 PNGs, metrics, NetCDF. Acc/F1: lag 0 0.771/0.811, lag 60 0.793/0.815. All-zero: neg 0 at lag 0, 29.5% (1502) at lag 60; pos 0. The share depends on the network |
| `LRP_compare_trained` (lags 0 60) | 0, 202 s | prob diff 8.3e-07 / 1.0e-06; all per-sample numbers 1.0000, **but at lag 60 `pearson_mean_map` = NaN for LRP-A1B0 (pos and neg): the Torch mean map is NaN at every wet point** (ncdump: 105248/105304 of 105300 points fill, vs 73493 = land for Keras). So some Torch samples have non-finite relevance; the nan functions hid them. Being checked by 3545869 |
| `OI_keras_trained`, `OI_keras_in_torch_trained`, `OI_torch_trained` | 0, 35 / 28 / 29 s | the KERAS_MODELS / TORCH_MODELS hand-off works, `--source saved-torch` works |

Changes made because of it (2026-11-01, untested until 3545869 finishes):
- `eccov4r5_negative_class_LRP.py` `difference_model`: `kernel_initializer`/`bias_initializer = 'zeros'`.
- `eccov4r5_LRP_A1B0_compare.py` `agreement`: also reports, per class and method, `keras` / `torch` `{n_all_zero, n_nonfinite}` and
  `n_pearson_nan`, since the nan functions hid the Torch NaN samples; the print line handles non-float values.

**Follow-up job 3545869** (`gpu-a100-dev`, 45 min, `--dependency=afterany:3545828`, script `scripts_TomsQoI/results/checks/check_followup.slurm`,
not in git; log `scripts_TomsQoI/results/checks/check_followup.3545869.out`, run dir `results/checks/followup_3545869/`): runs
`results/checks/check_torch_nan_map.py` (lag 60 of the mini Keras network: raw and normalized relevance of both libraries, how many samples are
non-finite in each, and for the first 5 bad Torch samples their logits and Keras sums), the compare script with the new counts on that network,
and the fixed `negative_class` on the saved models (lags 0 60, `--n-lrp 20`).

**Result of 3545869 (2026-11-01): all three exit 0** (362 / 117 / 42 s). Logs in `results/checks/followup_3545869/`.
- **The Torch NaN map is not a Torch bug.** Lag 60, mini Keras network: raw relevance Keras vs Torch max abs diff 1.4e-09 to 1.4e-06 (pos n
  3864, neg n 4989, both methods), normalized 3.7e-06 to 1.7e-05. Non-finite samples: pos 0 and 0; neg **122 in Keras and 122 in Torch**, the
  same samples, the all-zero ones (normalized 0/0 = NaN, 31811 NaN = every wet point, logits e.g. [-0.360, 0.193]). The new counts of the
  compare script say the same (`n_nonfinite` 122/122, `n_all_zero` 0/0 after normalization, `n_pearson_nan` 122) and it still gets
  `pearson_mean_map` NaN for LRP-A1B0 pos and neg (not for A1B0-B). So the NaN comes from `normalized_mean_relevance`
  (`eccov4r5_common.py:293`, divides each sample by its **max**, not its max abs). Hypothesis being checked by 3545911: with the bias kept
  (A1B0, not A1B0-B), a sample whose explained logit is negative gets relevance <= 0; its max is exactly 0 in one library (-> -inf / NaN)
  and a tiny rounding value in the other (-> huge finite values, wrong too). The earlier check found no negative explained logit for the
  saved models, but the 500-epoch networks may have some.
- **The negative-class mechanism of section 4.4 is wrong as stated.** Saved models (all samples, forward pass): `no_positive_contribution_share`
  = 0.000 at lags 0 and 60 for both classes, yet LRP on the first 20 lag 0 neg samples gave `all_zero_share` 1.000 for A1B0-B, A1B0, **and
  the logit-difference model** (so that remedy does not help). Lag 60 neg first 20: 0.000. Output bias: +0.187 (lag 0) / +0.194 (lag 60)
  for neg, -0.187 / -0.194 for pos; bias share of the logit median 0.10. Active neurons median (hidden layers 1, 2): neg [3, 5] / [3, 4],
  pos [5, 5] / [5, 7]. So the relevance is blocked in a lower layer. **Changed (2026-11-01)**: `mechanism()` in
  `eccov4r5_negative_class_LRP.py` now also traces which neurons carry A1B0 relevance down through every hidden layer (a neuron carries it
  if it has a positive contribution h_i w_ij > 0 into a carrying neuron above) and reports `blocked_share` (no carrying neuron in some
  layer), `blocked_share_by_layer`, `relevance_neurons_median`, `blocked_share_lrp_samples`. Expected: `blocked_share_lrp_samples` = the
  all-zero share of the same samples. Tested by 3545911.

**Follow-up job 3545911** (`gpu-a100-dev`, `results/checks/check_followup2.slurm`, log `results/checks/check_followup2.3545911.out`, run dir
`results/checks/followup2_3545911/`): `check_max_zero.py` (the max == 0 hypothesis, lag 60 of the mini network), the updated `negative_class`
on the saved models (lags 0 60, `--n-lrp 100`) and on the mini network.

**Result of 3545911 (ran 2026-11-01, all exit 0, read 2026-11-10).**
- `check_max_zero.py`, lag 60 of the mini Keras network, LRP-A1B0 (bias kept), normalized: pos n 3864, explained logit < 0 in 98; 96 samples
  have max relevance **exactly 0 in both libraries** (all of them with a negative logit; none with 0 < max < 1e-6). neg n 4989: logit < 0
  in 2479, max == 0 in 2465. The mean map of `normalized_mean_relevance` is then **inf at all 31811 wet points in Keras, NaN in Torch**
  (31755 NaN + 56 inf for pos). **Hypothesis confirmed**: a negative explained logit gives relevance <= 0, the per-sample division by the max
  (0) breaks the whole mean map in both libraries (the sign of the zeros differs, so Keras gets inf and Torch NaN). Only LRP-A1B0 (bias kept);
  LRP-A1B0-B (IB) is not affected. The saved notebook models have no negative explained logit (section 4.3), so the published maps are fine.
- `negative_class` with the layer-by-layer trace, **the mechanism is confirmed**:

  | network | lag, class | n | blocked_share (all samples) | blocked in layer [h1, h2] | blocked, first 100 | all-zero LRP, first 100 (A1B0-B, A1B0) | difference model all-zero |
  |---|---|---|---|---|---|---|---|
  | saved | 0 neg | 5770 | **0.255** | [0.255, 0.0] | 1.000 | 1.000, 1.000 | 1.000 |
  | saved | 60 neg | 5816 | **0.216** | [0.216, 0.0] | 0.000 | 0.000, 0.000 | 0.000 |
  | mini (3 ep.) | 0 neg | 4777 | **0.348** | [0.348, 0.345] | 0.000 | 0.000, 0.000 | 1.000 |
  | mini (3 ep.) | 60 neg | 4989 | **0.024** | [0.024, 0.021] | 0.000 | 0.000, 0.000 | 0.000 |
  | both | pos, 0 and 60 | | 0.000 | | | 0 | 0 |

  `no_positive_contribution_share` (only the last layer, the first hypothesis) is 0.000 for the saved models: the block is in **hidden layer 1**
  (no neuron of h1 has a positive contribution h_i w_ij > 0 into the neurons of h2 that carry relevance; median number of carrying neurons
  [1, 1] for saved neg vs [4-5, 4-7] for pos). The all-sample blocked shares equal the all-sample all-zero shares that LRP itself printed:
  mini lag 0 neg 0.348 = 1662/4777, lag 60 0.024 = 122/4989 (3545828), saved lag 0 neg 0.255 = 1472/5770 (`n_nonfinite` of 3544555). So
  the forward-pass trace predicts exactly which samples LRP leaves at zero. The logit-difference remedy does **not** help (all-zero 1.000
  where the original is 1.000, and on the mini lag 0 it makes 100 samples all-zero that were not), so it is not the fix.
  Output bias of the saved models: neg +0.187 / +0.194, pos -0.187 / -0.194 (lags 0 / 60).

### Results of the real jobs (all finished between 2026-11-02 and 2026-11-07; read 2026-11-10)

| job | state | time | notes |
|---|---|---|---|
| 3544554 OI saved lag 0 | COMPLETED | OI_keras 1748 s, OI_torch 76 s | Keras pos 0.817 -> 0.99823 after 1000 steps, Torch 0.99123; neg 0.98041 -> Keras 0.98513, Torch 0.98207 (the loss difference of the OI row above) |
| 3544555 LRP_compare saved lag 0 | COMPLETED | 1372 s | all Pearson 1.0000, prob diff 2.09e-06; neg: **1472 non-finite (all-zero) samples of 5770 in both** libraries (25.5%) |
| 3544556 negative_class | **TIMEOUT** (2 h) | | `negative_class.log` **empty**: Python block-buffers stdout to a file and the kill lost it. Not known how far it got. **Fixed 2026-11-10**: `run()` in `run_eccov4r5.slurm`, `run_eccov4r5_mini.slurm`, `run_easy_tests.slurm` now uses `python -u`. Needs a resubmit with a longer `-t` (6 h) |
| 3544560 LRP_keras 500 ep. | COMPLETED | 2500 s | metrics below |
| 3544561 LRP_torch 500 ep. | COMPLETED | 2715 s | metrics below |
| 3544562 LRP_compare_trained | COMPLETED | 1747 s | all lags: prob diff 1.2-2.3e-06, Pearson 1.0000 wherever defined; non-finite samples identical in both libraries (see table below); `pearson_mean_map` NaN where all samples are non-finite |
| 3544568 OI_keras_trained | **TIMEOUT** (6 h) | | finished lags -60..150, killed during lag 180. Keras OI is slow (about 0.12 s/step here, 0.87 s/step in 3544554) vs Torch (271 s for all 9 lags x 2 kinds x 10000 steps). Lag 180 needs a rerun, or Keras OI made faster |
| 3544569 OI_keras_in_torch_trained | COMPLETED | 290 s | |
| 3544570 OI_torch_trained | COMPLETED | 271 s | |

**Retrained networks, accuracy / F1 and all-zero LRP share (both methods give the same count) per lag** (from `metrics_<b>.json` and the
"Number of all-zero samples" lines of `ecco_3544560/LRP_keras.log`, `ecco_3544561/LRP_torch.log`):

| lag | Keras acc / F1 | Keras all-zero pos / neg | Torch acc / F1 | Torch all-zero pos / neg |
|---|---|---|---|---|
| -60 | 0.918 / 0.926 | 0 / 211 (3.7%) | 0.919 / 0.927 | 0 / 74 (1.3%) |
| -30 | 0.935 / 0.941 | 0 / **5832 (100%)** | 0.915 / 0.921 | 0 / 18 (0.3%) |
| 0 | 0.936 / 0.942 | 0 / **5836 (100%)** | 0.932 / 0.938 | 0 / 0 |
| 30 | 0.940 / 0.945 | 0 / 0 | 0.932 / 0.939 | 0 / 1570 (26.8%) |
| 60 | 0.926 / 0.933 | 0 / 58 (1.0%) | 0.926 / 0.933 | 0 / 1078 (18.6%) |
| 90 | 0.906 / 0.915 | 0 / 0 | 0.906 / 0.915 | 0 / 0 |
| 120 | 0.911 / 0.919 | **4464 (100%) / 5688 (100%)** | 0.913 / 0.920 | 0 / 29 (0.5%) |
| 150 | 0.943 / 0.947 | 0 / 6 (0.1%) | 0.946 / 0.950 | 31 (0.6%) / 0 |
| 180 | 0.937 / 0.942 | 53 (1.1%) / 0 | 0.933 / 0.939 | 0 / 0 |

Accuracy of the two backends agrees within 0.02 at every lag. **Three Keras networks (lags -30, 0 neg; 120 both classes) give no LRP
relevance at all for a whole class**: the relevance is blocked in a hidden layer for every sample, although they classify at 91-94%. The
OI confirms it is the network, not LRP: for exactly those cases the OI does not move in 10000 steps (zero input gradient) in Keras and in
the same networks run by Torch (`OI_keras_trained` / `OI_keras_in_torch_trained`: lag -30 neg 0.01707 -> 0.01707, lag 0 neg 0.05190 ->
0.05190, lag 120 neg 0.01952 -> 0.01952; lag 120 pos stops at 0.93305 in both). So for those classes the explained logit does not depend
on the input near the samples (dead ReLU paths: the class is decided by the other logit and the bias). The Torch-trained networks do not
show it (OI moves at every lag). Why Keras training produces them and Torch not (initializers: Keras glorot_uniform vs Torch kaiming_uniform;
optimizer details) is open. The published notebook networks have the milder version (25.5% at lag 0, 21.6% at lag 60).

OI end predictions (10000 steps, pos target [1,0] / neg target [0,1]): Keras-trained in Keras reaches 0.9999+ (pos) and 0.9993-0.9997
(neg, where it moves); the same networks in Torch reach 0.998-0.9995 / 0.993-0.996; Torch-trained 0.996-0.999 / 0.990-0.997. The
Keras vs Keras-in-Torch difference is the loss difference (OI row of the 3545828 table). The maps themselves have not been compared yet.

**To do next (2026-11-10)**: resubmit `negative_class` (`-t 06:00:00`, now unbuffered); rerun `OI_keras_trained` lag 180 or speed up Keras
OI; compare the OI and LRP maps of the three cases numerically (a job, no Python on the login node); write the script that compares the two
separately trained runs; owner decision still open: what to do about the dead-path networks (retrain with another seed / initializer, or
report it; my recommendation: first find why Keras training makes them and Torch not, then retrain).

**Owner decisions of 2026-11-10/11 and the changes made:**
- **OI loss: "the more principled thing"**, BCE of the softmax probabilities in both backends. The owner first asked whether the two losses
  are equivalent: they are not. With 2 classes Torch's BCE of the probabilities is -log sigmoid(z0 - z1) (depends only on the logit
  difference, gradient along grad(z0 - z1)); Keras' was ½[softplus(-z0) + softplus(z1)] (gradient -½(1-sigmoid(z0)) grad z0 +
  ½ sigmoid(z1) grad z1, another direction unless z1 = -z0). Change: `eccov4r5_OI_keras.py` `optimal_input` uses
  `keras_bce(y_true, tf.identity(y_pred))` (tf.identity drops the cached `_keras_logits`, so Keras clips the probabilities to
  [1e-7, 1-1e-7] and takes logs, as BCELoss, which clamps the log at -100). Torch unchanged. **Consequence: every Keras OI so far
  (3544554, 3544568, the mini runs, and the notebooks' published OI) used the old loss; they need rerunning.** Test: dev job **3573687**
  (`results/checks/check_oi_loss.slurm`, saved models, lags 0 60, both kinds, 200 steps; log `results/checks/check_oi_loss.3573687.out`,
  run dir `results/checks/oi_loss_3573687/`): Keras and Torch predictions should now agree at every printed step to about 1e-5.
  **Result (2026-11-11): passed.** Both exit 0 (Keras 412 s, Torch 35 s). At every printed step (0, 50, ..., 200), lags 0 and 60, both
  kinds, the Keras and Torch predictions agree to about 1e-7 (e.g. lag 60 pos step 200: 0.97754353 both; before the fix 20 steps already
  differed by 0.02). The losses are now the same.
- **VS Code server moved to /work (2026-11-11)**: the owner set `"remote.SSH.serverInstallPath": {"ls6.tacc.utexas.edu":
  "/work/07665/shrey911/ls6"}` on the laptop (ssh host alias `ls6.tacc.utexas.edu`). The new server runs from
  `/work/07665/shrey911/ls6/.vscode-server` and this Claude Code session resumed (same transcript `3cf0f372-...jsonl`). The Machine
  settings file was copied there. The old `~/.vscode-server` (about 1.7 GB) can be deleted once no process runs from it.
- **Map normalization: divide by the max abs** (owner: "I also like dividing by max absolute value"). Change: `normalized_mean_relevance`
  in `eccov4r5_common.py` (used by the LRP scripts' maps and NetCDF, the compare script and `negative_class`). Note that the scripts
  already normalize each sample by `NORMALIZE = MaxAbs` in `XAIKeras`/`XAITorch` before this, so for them the division is by 1 except
  for all-zero samples (still NaN, left out). Not rerun yet; the LRP-A1B0 maps of the 500-epoch runs at the lags with a negative explained
  logit are broken in the existing output (inf / NaN) and need the LRP scripts rerun.
- **/home1 quota**: it was at 98% (9.8 of 10 GB, mostly `~/.vscode-server` 5.8 GB and `~/.cache/vscode-cpptools` 2.5 GB), and sbatch
  refuses to submit above 90%. The owner said **not to touch anything of VS Code** (this session runs in it). Instead the 1.1 GB desktop
  trash `~/.local/share/Trash` (the owner's 2024 LRP NetCDF results, e.g. `LRP_A1B0_newAnomalies_allSignsFlipped.nc`,
  `LRP_A1B0_IB_B_3sigma_...nc`, `LRP_Z.nc`; maybe useful for the negative-class question) was **moved** (copied, verified with
  `diff -rq`, then removed) to `/work/07665/shrey911/ls6/home_trash_backup_2026-11-10/`. /home1 now 8.8 GB (87.6%): little headroom.
  **2026-11-11, with the owner's approval**: deleted the unused VS Code items (a 2.6 GB C/C++ symbol database of April 2024 in
  `~/.cache/vscode-cpptools`, four older servers in `~/.vscode-server/cli/servers` (kept the running `Stable-dcd3ad6e...`), the legacy
  `~/.vscode-server/bin/` of 2024, `data/CachedExtensionVSIXs`, four old `code-*` launchers). /home1 now **2.9 GB (28.7%)**. The C/C++
  extension stays (the Fortran linter needs it); its caches now go to `/work/07665/shrey911/ls6/.cache/vscode-cpptools`, capped at
  512 MB, via `~/.vscode-server/data/Machine/settings.json` (new). Suggested to the owner, on the laptop (I cannot change it):
  `"remote.SSH.serverInstallPath": { "<ssh host alias>": "/work/07665/shrey911/ls6" }` so future servers install on /work (the
  launcher keeps the last 5 versions, about 650 MB each), and optionally `"update.mode": "manual"`. If that is done, copy the
  Machine settings file to `/work/07665/shrey911/ls6/.vscode-server/data/Machine/`. Claude Code sessions are in `~/.claude/projects/`
  and are not affected by the install path.

### 2026-11-11: dead Keras networks, Torch -> Keras, Keras OI speed (owner's requests)

The owner: "Keras networks that are dead seem suspicious. Investigate that. Also, have more code in the repo that trains on Torch and
imports to Keras and uses XAI. Will be good to compare." and "Also investigate why Keras OI is slower than Torch OI."

**What the logs already say (read on the login node, no Python):**
- Not an early checkpoint: in `ecco_3544560/LRP_keras.log` the last "val_loss improved" is at epoch 468-500 for every lag (-30: 468,
  120: 475, the others 489-500). So the training had not converged at 500 epochs either.
- The dead Keras networks trained worse: final Keras val_loss per lag -60..180: 0.3007, **0.3039**, **0.3174**, 0.2781, 0.2927, 0.2947,
  **0.3560**, 0.2925, 0.2896 (bold = the dead ones; lag 120, both classes dead, is clearly the worst). Torch at epoch 490: 0.281-0.307 at
  every lag. First epoch: Keras val_loss 0.99-1.87, Torch 1.03-2.34, train loss about 8.5 (the L2 penalty 10 of the first layer dominates).
- Same architecture and settings in both backends: 8-8-2 ReLU/softmax, He-normal (Torch truncates at 2 std with the Keras constant,
  `Trainer.py:371-376`), zero biases, SGD lr 0.01 Nesterov momentum 0.9, `step_decay` (halves every 50 epochs), batch 128, class
  weights, L2 10 (layer 1, weights and biases) / 0.01 (layers 2, 3), best val_loss checkpoint. Different: the random streams (init,
  shuffling), and **Keras trains in TF1 graph mode** (the Keras LRP script calls `tf.compat.v1.disable_eager_execution()` for innvestigate
  before training; the log shows the v1 format "7233/7233 - 0s - loss ... - 58us/sample") **with TF32 on** (not disabled in that script).
- Hypothesis: with only 8 neurons in layer 1 and a strong L2 on them, some ReLUs die early and never recover; whether that is chance or
  systematic for Keras is what the seed study below answers.

**New code (2026-11-11, untested until 3573699 runs):**
- `src/XAIRT/utils.py`: `torch_to_keras(model: nn.Sequential) -> keras Model`, the reverse of `keras_to_torch` (Linear + optional
  ReLU/Sigmoid/Tanh/Softmax(dim -1 or 1) modules -> Dense layers with the activation; weights transposed; zero initializers, set right
  after, for the same "Random ops require a seed" reason as `difference_model`). Exported in `__all__`.
- `eccov4r5_LRP_A1B0_keras.py` `make_get_model`: new `--source saved-torch` (load `model{lag}_noL1.pt` with
  `TrainTorchFullyConnectedNN.load_saved_model`, then `torch_to_keras`, then the same `saved_` layer renaming). Added to the sources of
  `eccov4r5_OI_keras.py`, `eccov4r5_analyze_keras.py`, `eccov4r5_LRP_A1B0_compare.py` (Torch-trained network explained by both libraries;
  the Torch side is `keras_to_torch` of the Keras copy, an exact round trip; docstring updated) and `eccov4r5_negative_class_LRP.py`.
  `--source` help text in `eccov4r5_common.py` updated.
- `run_eccov4r5.slurm`: new runs `LRP_compare_torch_trained`, `OI_torch_in_keras_trained` (need `TORCH_MODELS`), `dead_neurons_seeds`,
  `dead_neurons_keras_eager`. `run_eccov4r5_mini.slurm`: miniatures of the first two.
- New `eccov4r5_dead_neurons.py` (docstring has the details): `--inspect LABEL=DIR` (existing .h5 / .pt networks) or train mode
  (`--backends --seeds --epochs-list --lags`, trained with each backend's `make_train_fn`, as the LRP scripts). Per network: per-neuron
  active share of the samples, `n_dead` per layer, first-layer biases and weight norms, and per class `constant_share` (correct samples
  whose logit has no active path from the input: zero input gradient, so no OI movement and zero relevance for any rule) and
  `blocked_share` (A1B0, `mechanism()` of the negative-class script). `--keras-eager`, `--no-tf32` train Keras differently from the LRP
  script. Writes `dead_neurons.json` (after every trained network, so a timeout keeps what was done).
- One-off `results/checks/check_oi_speed.py` (not in git): times 200 OI steps of the saved lag 0 network (pos), Keras as the script does it
  (one `tf.function` call per step from Python with a numpy round trip), with the tensors kept on the device, and the whole loop in one
  graph, each on GPU and CPU (a new `tf.function` per device, since a trace keeps its device placement), plus Torch on CPU (as
  `eccov4r5_OI_torch.py`, which **never moves to the GPU**) and GPU; `--mode script` (op determinism on, as the script) and
  `--mode nodeterminism`. Prints ms/step and the max difference of the final input from the script's way. Suspects: per-step Python
  dispatch + host<->GPU copies on a network this small (where the CPU wins), and the deterministic kernels.

**Jobs (2026-11-11):**

| job | partition | what | waits for |
|---|---|---|---|
| 3573687 | gpu-a100-dev, 30 min | OI loss check (above) | - |
| 3573699 | gpu-a100-dev, 1.5 h | `results/checks/check_followup3.slurm`: `check_oi_speed.py` both modes; `dead_neurons --inspect` of saved, keras500 (3544560), torch500 (3544561) at all 9 lags and of the mini 3-epoch networks; `dead_neurons` train mode test (3 epochs, lag 0, both backends); `LRP_compare_torch_trained`, `OI_torch_in_keras_trained`, `OI_torch_trained` on the mini Torch networks (lags 0 60, 20 OI steps: Keras and Torch OI of the same Torch network, which with the new loss should agree). **Exits 1 if any run fails.** Log `results/checks/check_followup3.<id>.out`, run dir `results/checks/followup3_<id>/` | 3573687 `afterany` |
| 3573700 | gpu-a100, 8 h | `run_eccov4r5.slurm dead_neurons_seeds dead_neurons_keras_eager`: Keras and Torch, seeds 42 43 44, lags -30 0 30 60 120, 500 epochs; then Keras eager + no TF32, seeds 42 43 44, lags -30 0 120. Seed 42 should reproduce 3544560/3544561 (a check of determinism). About 4 h estimated. Output `results/ecco_3573700/dead_neurons_*/dead_neurons.json` | 3573699 `afterok` (if 3573699 fails it stays `DependencyNeverSatisfied`: fix, `scancel`, resubmit) |

The three cases the owner asked for, for LRP and for OI: **Keras** (3544560 LRP, 3544568 OI), **Torch** (3544561 LRP, 3544570 OI), **Keras network
with Torch XAI** (3544562 LRP, 3544569 OI).

Why 3544561 waits for 3544567: 3544561 imports the refactored `src/XAIRT/Trainer.py` (`load_saved_model`), and 3544567 is the quick test that
training and reloading a Torch network still works. It was first put on hold and then given the dependency, so that it starts by itself if the test
passes, without anyone having to release it.

**If a job whose dependency failed stays pending with reason `DependencyNeverSatisfied`**: read the failed job's `.out`/`.err` in
`scripts_TomsQoI/results/logs/` and the per-run `.log` in its run directory, fix, `scancel` the stuck job, and resubmit (with the dependency or the
models directory of the new run).

What to check, in order, and what to expect:

1. **3544567** (`results/logs/XAIRT_easy.3544567.out`): **passed** before the session ended: both runs `exit 0`, no tracebacks, Keras vs Captum
   `sign_split` 2.4e-07 and Captum `BoundedRule` vs numpy 1.6-1.8e-07 as before; 3544561 then lost its dependency and waits for a node. If `LRP_torch` fails in `loadBestModel`/`load_saved_model`, the refactor
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

**Correction (2026-11-01, mini job 3545828): the first 400 are not representative.** The published file has **no NaN at any wet point in any
map**, the lag 0 negative maps included (`negative_class.log` of the mini job, share 0.000 for every variable), so at lag 0 some negative samples
do get relevance. And `LRP_compare` on all 5770 lag 0 negative samples gave finite per-sample Pearson values (it leaves NaN samples out, so it
does not say how many are all-zero). The 100% / 28% above are only for the first 400 samples in the order of `correct_indices` (probably
time order, so the start of the record). The true share over all samples is the `no_positive_contribution_share` of `negative_class.json`
(forward pass on all samples). Note: `--n-lrp` also takes the first samples, so `all_zero_share_*` there has the same bias; compare it with
`no_positive_contribution_share_lrp_samples`, not with the all-sample share. The published maps are then averages over the explained subset
only, not NaN; whether that biases them is still the open question.

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
  (`--source saved-torch`). **Tested by job 3544567: passed** (training and reloading a Torch network through `loadBestModel` works).

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
- New `run_easy_tests.slurm`, `run_eccov4r5.slurm` (section 2.5). New on 2026-11-01: `run_eccov4r5_mini.slurm` (section 2.5).

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
