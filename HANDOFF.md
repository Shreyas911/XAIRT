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

Sessions so far: `3cf0f372-6954-4299-afcf-169612297e29` (2026-10-31), `8925f256-4efe-4bf8-ba41-33085efeed30` (2026-11-14, started
from `~`, so its transcript is in `~/.claude/projects/-home1-07665-shrey911/`, not the XAIRT folder; backed up to
`claude_session_backups/home_project/`). Add each new session id here. A session can be resumed with
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

### Queued 2026-11-11, before the owner's week (or more) away: everything discussed, as jobs

The owner: "Are all the things we discussed being tested? I will be gone for a week or more so want to start as many jobs as
possible." Submitted 2026-11-11 (3573699 was running then). `K` = `results/ecco_3544560/LRP_keras/models` (500-epoch Keras networks),
`T` = `results/ecco_3544561/LRP_torch/models` (500-epoch Torch networks). All `run_eccov4r5.slurm` jobs log to
`results/logs/XAIRT_ecco.<id>.out` (one line per run: exit code, seconds) and write to `results/ecco_<id>/<run name>/` and
`<run name>.log` (now unbuffered). Limits found: gpu-a100 max 48 h per job, 20 running and 100 submitted jobs per user;
**gpu-a100-dev at most 3 submitted jobs per user** (`QOSMaxSubmitJobPerUserLimit`).

New run names in `run_eccov4r5.slurm`: `OI_keras_saved_full`, `OI_torch_saved_full` (saved notebook networks, notebook OI settings: 9
lags, both kinds, 10000 steps), `LRP_compare_saved_all` (saved networks, 9 lags, both libraries, new normalization),
`negative_class_trained` (KERAS_MODELS), `negative_class_torch_trained` (TORCH_MODELS, via torch_to_keras). New script
`eccov4r5_compare_maps.py` (compares the maps of two NetCDF files, variable by variable: Pearson over points finite in both, NaN/inf
counts, max abs diff; `--pair LABEL|A|B[|SUFFIX_A|SUFFIX_B]`, missing files skipped; no ECCO data read). One-off
`results/checks/compare_all_maps.slurm` (not in git) lists all the pairs below.

| job | partition, limit | runs | what it answers | waits for |
|---|---|---|---|---|
| 3573699 | dev, 1.5 h | check_followup3 (above) | OI speed, dead neurons of all networks, first runs of the new code | (was running) |
| 3573700 | gpu-a100, 8 h | dead_neurons_seeds, dead_neurons_keras_eager | dead networks: Keras or seed? graph mode / TF32? | 3573699 ok |
| 3573738 | dev, 1 h | `run_eccov4r5_mini.slurm` (all miniatures, now with the Torch->Keras runs) | every run still works end to end | 3573699 any |
| 3573740 | dev, 30 min | compare_all_maps (first) | tests `eccov4r5_compare_maps.py` on what exists: Keras500 vs Torch500 LRP maps (old normalization), old OI | 3573738 any |
| 3573727 | gpu-a100, 24 h | negative_class (saved, 9 lags, n-lrp 300) | the 3544556 rerun (that one timed out at 2 h, log lost) with the layer-by-layer `blocked_share` | - |
| 3573729 | gpu-a100, 24 h | negative_class_trained (K) | the same on the 500-epoch Keras networks (lags -30, 0, 120 are 100% blocked?) | - |
| 3573730 | gpu-a100, 24 h | negative_class_torch_trained (T) | the same on the 500-epoch Torch networks | 3573699 ok |
| 3573731 | gpu-a100, 10 h | LRP_compare_saved_all | published maps vs both libraries, all 9 lags, new normalization (LRP-A1B0-B should match the published file closely) | - |
| 3573732 | gpu-a100, 6 h | LRP_compare_trained (K) | rerun of 3544562 with the new normalization (no inf/NaN maps) | - |
| 3573733 | gpu-a100, 6 h | LRP_compare_torch_trained (T) | Torch-trained networks explained by innvestigate and Captum (the owner's "train on Torch, import to Keras, XAI") | 3573699 ok |
| 3573734 | gpu-a100, 36 h | OI_keras_saved_full | Keras OI, new loss, saved networks, notebook settings (slow: 3544568 needed >6 h for 9 lags) | - |
| 3573735 | gpu-a100, 4 h | OI_keras, OI_torch, OI_torch_saved_full | new-loss Keras OI lag 0 1000 steps (vs old 3544554), Torch OI with notebook settings | - |
| 3573736 | gpu-a100, 36 h | OI_keras_trained (K) | rerun of 3544568 (timed out, old loss) with the new loss | - |
| 3573737 | gpu-a100, 36 h | OI_torch_in_keras_trained (T) | Keras OI of the Torch-trained networks | 3573699 ok |
| 3573742 | gpu-a100, 30 min | compare_all_maps (final) | all pairs: published vs saved (Keras, Torch); Keras vs Torch per file; Keras500 vs Torch500 (new normalization); OI old vs new loss; OI Keras vs Torch for saved, Keras500, Torch500; Keras500 vs Torch500 OI. Log `results/checks/compare_all_maps.<id>.out`, results `results/checks/maps_<id>/compare_maps.log` and `.json` | all of 3573727-3573737 (any) |

What to look at first when back: `squeue -u $USER`; `sacct -j 3573699,3573700,3573727,3573729,3573730,3573731,3573732,3573733,3573734,3573735,3573736,3573737,3573738,3573740,3573742 -X -o JobID,State,Elapsed,ExitCode`;
then `results/checks/maps_3573742/compare_maps.log` (one line per map), `results/checks/check_followup3.3573699.out`
and its `followup3_3573699/oi_speed_*.log`, `dead_inspect.log`; `results/ecco_3573700/dead_neurons_*.log`. If 3573699 failed, the four
jobs that wait for it `afterok` (3573700, 3573730, 3573733, 3573737) stay pending with `DependencyNeverSatisfied`: fix, `scancel`,
resubmit them (commands in this table). Not covered by any job: the Keras OI speed-up (needs the 3573699 timings first) and the old
`~/.vscode-server` deletion.

**3573699 passed (2026-11-11 00:36, all 8 runs exit 0)**, so 3573700, 3573730, 3573733, 3573737 are released (now waiting for nodes).
Its Keras OI speed result (`followup3_3573699/oi_speed_{script,nodeterminism}.log`, 200 steps, saved lag 0 network, pos), ms/step:

| way | GPU, determinism on | CPU, on | GPU, off | CPU, off |
|---|---|---|---|---|
| Keras as the script (tf.function call per step, numpy round trip) | 0.716 | 0.770 | 0.769 | 1.073 |
| Keras, tensors kept on the device | 0.375 | 0.577 | 0.378 | 0.933 |
| Keras, whole loop in one graph | 0.237 | 0.231 | 0.272 | 0.624 |
| Torch as its script (CPU) | 0.275 | | 0.303 | |
| Torch on the GPU | 0.518 | | 0.532 | |

All ways give the same final input (max diff 3e-7, 0 on the same device). Determinism does not slow it down. **So the gradient step is
not why the Keras OI script is slow**: 0.7 ms/step would be 0.6 s for the 800 steps of 3573687, which took 412 s in Keras (35 s Torch);
3544568 took about 0.1 s/step over 10000-step runs. Something else in the script costs time, and more with more steps (the mini runs
with 20 steps were as fast as Torch). Main suspect: `model.predict` in the loop (every `--print-freq` steps). **Profiling job 3573842**
(`gpu-a100-dev`, `results/checks/check_oi_profile.slurm`, not in git): cProfile of the real `eccov4r5_OI_keras.py`, lag 0, pos, 400
steps, with `--print-freq 50` and `100000` (no prints); output `results/checks/oi_profile_3573842/prof_pf*.txt` (top functions by
cumulative and own time) and the run times in `check_oi_profile.3573842.out`.

### Multi-field input (2026-11-11, owner: "Modify the code so that more than one input ECCO fields can be handled in the future")

Done on a **separate branch `multi_field` in a separate git worktree, `/work/07665/shrey911/ls6/XAIRT_multi_field`**, because the 11
queued jobs read `scripts_TomsQoI/*.py` of this checkout when they start (any time during the owner's week away); editing them here
could break those runs. Merge `multi_field` into `clean_keras_torch` after the queued jobs have started (or finished) and the test
below passed: `git merge multi_field` here, then `git worktree remove /work/07665/shrey911/ls6/XAIRT_multi_field`. Nothing is
committed on the branch yet (the owner commits).

Design (all in the worktree, `scripts_TomsQoI/`):
- `eccov4r5_common.py`: `FIELDS = {'SST': load_sst}`, a registry of loaders returning daily (days, 13, 90, 90) fields on the same days
  as `SST_all.nc`. `--fields` (default `SST`) and `--field-scale none|std` (default `none`) in `parse_args`, so every script has them.
  `load_anomalies(r4, r5, fields, field_scale)` anomalizes each field at the SST wet points and puts them side by side:
  `X[:, k*n_wet:(k+1)*n_wet]` is field k. **All fields share the SST wet-point mask** (surface fields on LLC90 only; a field with its
  own mask would need a mask per field). The QoI stays the SST anomaly at [10, 1, 43]. `field_scale 'std'` divides each field by its
  std over all wet points and days (all days, so the test period is included in the std: negligible leakage, but note it), needed when
  fields have different units, also for the Bounded rule's (-1, 1) bounds. `field_maps(values, wetpoints, fields)` splits a
  (n_fields*n_wet,) vector into per-field LLC maps with a tag: `''` for a single field (so **every NetCDF variable and figure name of the
  single-SST experiment is unchanged**) and `'<FIELD>_'` otherwise (e.g. `lrp_0_SSH_a1b0_b_pos_all`, `OI_SSH_0`, `SSH_composite_pos_keras.png`).
  `field_tags(fields)` lists the tags. `make_context` carries `ctx.fields`; `run_experiment` / `finalize` (LRP scripts) write per-field
  composites, relevance maps and figures.
- `eccov4r5_OI_common.py` (OI scripts) and `eccov4r5_LRP_A1B0_compare.py`: per-field maps and figures the same way.
- Unchanged, and fine with several fields: `eccov4r5_dead_neurons.py`, `eccov4r5_analyze_*`, `eccov4r5_negative_class_LRP.py` (they
  use only X; the published-file NaN check uses the SST wet points). Unchanged and **SST only**: `eccov4r5_dataReader.py`,
  `eccov4r5_viz_common.py`, `eccov4r5_results_viz*.py` (SST diagnostics; they ignore `--fields`).
- To add a field: write `load_<name>(r4_dir, r5_dir)` and add it to `FIELDS`; run any script with `--fields SST <NAME> --field-scale std`.
  Networks trained on one set of fields only work with the same `--fields` (the input width differs).

**Test job 3573843** (`gpu-a100`, 1.5 h, submitted from the worktree; `results/checks/test_multi_field.slurm` and the driver
`results/checks/with_test_field.py` there, not in git; log `<worktree>/scripts_TomsQoI/results/checks/test_multi_field.3573843.out`,
run dir `.../results/checks/multi_3573843/`): (1) SST only, deterministic runs on the saved models (OI_torch 20 steps lags 0 60,
LRP_compare lag 0) with the main code and the branch, then `eccov4r5_compare_maps.py`: **every map must have pearson 1 and max|A-B| 0**
(`same_as_main.log`); (2) `--fields SST SST2 --field-scale std` (SST2 = a copy of SST registered by the driver): LRP_keras and
LRP_torch (3 epochs, lags 0 60), compare on both trained sets, Keras and Torch OI and negative_class on the Torch networks,
dead_neurons inspect: all must exit 0 and write `SST_...` and `SST2_...` maps. Exits 1 if anything failed.

The branch was committed (`3fe7653`, "Support several ECCO input fields (--fields, --field-scale), per-field maps") and **pushed by the
owner** (`origin/multi_field`; my shell has no GitHub credentials: the remote is HTTPS, no `gh`, GitHub not in `known_hosts`, so pushes
are the owner's). How the worktree works (the owner asked): `git worktree add -b multi_field <dir> HEAD`; the worktree's `.git` is a
one-line file `gitdir: /home1/07665/shrey911/XAIRT/.git/worktrees/XAIRT_multi_field`; that directory holds its `HEAD`, `index`,
`gitdir`, `commondir`; commits, branches and remotes are shared with this checkout. `git branch` marks a branch checked out in another
worktree with `+` (`*` = this folder). `git worktree list` shows both. Remove with `git worktree remove <dir>` (not `rm`; if deleted by
hand, `git worktree prune`). A branch can be checked out in only one worktree at a time.

### State at the end of 2026-11-11 (00:55), the owner away for a week or more

All 16 jobs pending, none started, none failed: gpu-a100 3573700, 3573727, 3573729-3573737, 3573742 (final map comparison, after all),
3573843 (multi-field test, from the worktree); gpu-a100-dev 3573738 (mini), 3573740 (maps, first), 3573842 (OI profile). gpu-a100 had
about 430 pending jobs on 2026-11-01 and ours waited 1-6 days then, so expect days.

**Order of work when the owner is back:**
1. `sacct` of all ids (command above); for any FAILED/TIMEOUT read `results/logs/XAIRT_ecco.<id>.out`, the `<run>.log` in
   `results/ecco_<id>/`, fix, resubmit. Jobs blocked by a failed `afterok` show `DependencyNeverSatisfied`.
2. Keras OI speed: `results/checks/oi_profile_3573842/prof_pf{50,100000}.txt`. If `model.predict` (or anything outside the gradient)
   dominates, fix `optimal_input` in `eccov4r5_OI_keras.py` (e.g. print with `model(x)` instead of `model.predict`, keep x as a tensor,
   or run the loop in one `tf.function`; `check_oi_speed.py` showed all these give the same result). Do it **after** the queued Keras OI
   jobs (3573734, 3573735, 3573736, 3573737) have started, or on the `multi_field` worktree.
3. Dead networks: `results/ecco_3573700/dead_neurons_seeds.log` and `dead_neurons_keras_eager.log` (per network: dead neurons,
   constant/blocked shares). Seed 42 should reproduce 3544560/61. Keras dead at other seeds too and Torch not -> a Keras training issue
   (then graph mode / TF32 via the eager run); dead for some seeds of both -> chance (8 neurons, L2 10). Also
   `results/checks/followup3_3573699/dead_inspect.log` (already there, not read yet: dead neurons of saved, keras500, torch500).
4. Maps: `results/checks/maps_3573742/compare_maps.log`. Key numbers: published vs saved (new normalization) for LRP-A1B0-B (should be
   ~1; if not, the normalization or the rules changed the published maps); OI old vs new loss; Keras vs Torch everywhere (should be ~1
   for the same network); Keras500 vs Torch500 (the networks differ, so this measures how robust the maps are to the training).
5. Negative class: `results/ecco_357372{7,9}/negative_class*.log`, `results/ecco_3573730/...`: `blocked_share` per lag and class for
   saved, Keras500, Torch500 networks.
6. Multi-field: `<worktree>/scripts_TomsQoI/results/checks/test_multi_field.3573843.out`; if passed and the queued jobs have started,
   merge `multi_field` and remove the worktree.
7. Open owner decisions: what to do about dead-path networks (after 3); whether the draft's OI needs the loss note (after 4); and the
   still-open negative-class remedy (the logit-difference idea did not work).
8. Leftovers: delete the old `~/.vscode-server` on /home1 (about 1.7 GB) once no process runs from it (`ps -u $USER -o args | grep
   /home1/07665/shrey911/.vscode-server` empty); reinstall VS Code extensions on the new /work server if not done (Python, Pylance,
   Python Debugger, Python Environments, C/C++, Fortran linter).

### 2026-11-14 (session started from `~`, so it did not load this project's `memory/`; start Claude from the repo root)

- `sacct`: 3573699, 3573738 (mini, all 12 runs exit 0), 3573740, 3573842, 3573843 COMPLETED exit 0. The other 12 gpu-a100 jobs
  (3573700, 3573727, 3573729-3573737, 3573742) still PENDING (Priority), 380 jobs pending in gpu-a100. `multi_field` not merged yet (the
  queued jobs have not started). 3573843 passed: `same_as_main.log` all maps pearson 1, max|A-B| 0; all two-field runs exit 0.
- **`followup3_3573699/dead_inspect.log` read: the "dead" Keras networks have no dead neurons.** keras500 lags -30, 0, 120: `dead [0, 0]`,
  h1 active shares 0.41-0.51, h2 0.71-0.88 for every neuron; yet `blocked_share` 1.000 (neg at -30 and 0, both classes at 120) and
  `constant_share` 0.182 (-30 neg), 0.557 (0 neg), 0.595 / 0.187 (120 pos / neg). So the dead-ReLU hypothesis of 2026-11-11 is wrong: the
  neurons fire, but no active last-hidden neuron has a positive contribution into that logit. Saved networks, neg blocked share per lag
  -60..180: 0.063, 0.003, 0.255, 0.265, 0.216, 0.171, 0, 0, 0. Torch500: blocked only at lag 30 (0.268) and 60 (0.186) neg.
- **3573842 profile read: inconclusive.** ~160 of 165 s of the print-freq-50 run was cold-node startup (`create_dynamic` 68 s, `dlopen` 15 s,
  NetCDF `getitem` 25 s, `TF_ListPhysicalDevices` 14 s); the print-freq-100000 run (same node, warm) took 27 s. The 400 steps cost seconds in
  both, so `model.predict` is not shown to be the problem at this size. **Follow-up job 3582278** (`gpu-a100-dev`,
  `results/checks/check_oi_profile2.slurm`, not in git): warm-up run, then 400 and 3000 steps without prints and 3000 with print-freq 100;
  log `results/checks/check_oi_profile2.3582278.out`, profiles `results/checks/oi_profile2_3582278/prof_*.txt`. If the 3000-step run costs
  much more than 7.5x the 400-step one per step, something in the loop accumulates.
- `maps_3573740/compare_maps.log` (Keras500 vs Torch500, old normalization): LRP-A1B0-B pos r 0.90-0.98 (lag 120 NaN: Keras blocked);
  neg 0.56 (60), 0.69 (30), 0.73 (-60), 0.91-0.97 (90-180), NaN at -30, 0; composites >= 0.994; OI pos 0.94 (120), 0.988-0.995 elsewhere;
  saved lag 0 OI old-loss Keras vs Torch r 0.988.
- **'forcing' dataset (owner, 2026-11-14)**: a colleague (D. Apte) trains a CNN on precomputed wind-stress anomalies with the AMOC
  at RAPID as the QoI (`/work/10027/dhruvapte26/vista/eccoXAI/sample_LRP_grid_improved_v2.py`). Owner: **one dataset at a time, never
  combined with SST**; keep our frameworks; add AMOC as the QoI of the new dataset; check the grid index offset; measure how much
  re-anomalizing the precomputed anomalies matters. Their files (read from the script, the folder itself was **not readable**:
  `eccoXAI` lacks group read; the colleague must `chmod -R g+rX` it): `INPUT_FORCINGS/eccov4r5_7daylpf_daily_anomaly_about_366dayclimatology_{oceTAUE,oceTAUN}_Jan1992_Dec2021.data`
  (>f4, (10958 days, n_wet), wet points only), `INPUT_FORCINGS/eccov4r5_iwetC2d.data` (>f4, compact (1170, 90) position of each
  column), `INPUT_FORCINGS/rac_wet.bin` (>f8, cell areas), `TARGET_QoIs/..._RAPID_QoI_Jan1992_Dec2021.data` (>f4, (10958,)). 1992-2021,
  not 2022. **Suspected bug in their placement**: their mask agrees with hFacC > 0 on only ~97.4% of cells, scattered, equal counts
  both ways, hFacC = 1 there: the signature of 1-based positions used as 0-based (every value one cell off). Not verified.
  - Code (on `multi_field`, **uncommitted**, compiles, **never run**): `eccov4r5_common.py`: `--dataset sst|forcing` (default sst,
    the SST path unchanged), `--forcing-dir` (default `/work/07665/shrey911/ls6/LRP_eccov4r5_forcing_data`, a copy to make, see
    below), `--forcing-anomalies as-is|reanomalize` (default as-is), `--fields` default per dataset (`DEFAULT_FIELDS`;
    forcing: oceTAUE oceTAUN, any `<name>` of the file pattern works). `forcing_wetpoints` converts an index map once with
    `ecco.llc_compact_to_tiles` (not the data) and **raises if any column lands on a dry hFacC cell**; `forcing_index_base`
    detects 0- vs 1-based (the one whose positions are all wet in compact hFacC; raises if neither) and prints it. Forcing wet points are their own (all latitudes, no YC > -20). `load_forcing`;
    `anomalize_new(..., num_years = 30)` for reanomalize (X and QoI). `class_weights(oneHotCost, n_test)`: SST keeps `N_TEST`,
    forcing uses its own test size at the same `TEST_SPLIT_FRAC`. `run_experiment` now uses `make_context` (one data path).
    `eccov4r5_negative_class_LRP.py`: the published-SST-map check is skipped for forcing.
  - Jobs (submitted later the same day, see the table below): `results/checks/check_forcing_data.py` (worktree; base 0 vs 1 against
    hFacC and against RAC vs rac_wet; variance anomalize_new removes, split into calendar-day mean and trend, per point; QoI class
    flips), `results/checks/test_forcing.slurm` (gpu-a100 2 h: the check, SST old-vs-new identity incl. 3-epoch training,
    every script with `--dataset forcing` both anomaly options), `results/checks/forcing_anomalies_full.slurm` (gpu-a100 16 h,
    afterok of the test: Torch, 9 lags, 500 epochs, as-is vs reanomalize, metrics and map correlations).
  - `test_forcing.slurm` now copies the five files into `LRP_eccov4r5_forcing_data/{INPUT_FORCINGS,TARGET_QoIs}` itself (once;
    `cp -p`, partial copies removed) and `cmp`s their `GRID/hFacC.data` with ours; **if the colleague's folder is still unreadable
    when it starts, it exits 1 at once** and 3582379 stays `DependencyNeverSatisfied`: get access, `scancel 3582379`, resubmit
    both from the worktree (`cd /work/07665/shrey911/ls6/XAIRT_multi_field; T=$(sbatch --parsable
    scripts_TomsQoI/results/checks/test_forcing.slurm); sbatch --dependency=afterok:$T scripts_TomsQoI/results/checks/forcing_anomalies_full.slurm`).

### Queued 2026-11-14, the owner away again ("queue as many jobs as you want; maybe tackle some open questions")

All logs in `scripts_TomsQoI/results/checks/` (main checkout unless noted). None of the new scripts has run yet.

| job | partition, limit | what | answers | waits for |
|---|---|---|---|---|
| 3582278 | dev, 1.5 h | `check_oi_profile2.slurm`: Keras OI, warm-up, then 400 / 3000 steps without prints, 3000 with print-freq 100 (cProfile) | Keras OI slowness: does the per-step cost grow with the steps? Out `check_oi_profile2.3582278.out`, `oi_profile2_3582278/prof_*.txt` | - |
| 3582364 | dev, 45 min | `rules_all_lags.slurm 0 --max-samples 50` into `rules_test/` | first run of `rules_all_lags.py` | - |
| 3582365 | dev, 20 min | `rules_all_lags.slurm --summarize` of `rules_test/` | the summary/plot mode works | 3582364 ok |
| 3582366-68 | gpu-a100, 8 h | `rules_all_lags.py` on the saved networks, lags -60 -30 0 / 30 60 90 / 120 150 180, all correctly predicted samples, into `rules_all_lags/` | (1) appendix bounds: A1B0-B (-1,1) vs training range vs no input rule, over all samples and lags; (2) negative class: do LRP-Z, epsilon, A2B1-IB, gradient x input give relevance where A1B0-B is all zero (`all_zero_share_on_a1b0_b_zero`), and does their mean map resemble the class composite (`corr_composite`)? | 3582365 ok |
| 3582369 | gpu-a100, 1 h | `--summarize` of `rules_all_lags/`: tables per class and statistic in `rules_all_lags.3582369.out`, one figure per method and class (`<method>_<cls>.png`, 9 lags) | | 3582366-68 any |
| 3582371-73 | gpu-a100, 4 h | `seeds_lrp.slurm keras 42/43/44`: `eccov4r5_LRP_A1B0_keras.py --seed`, full settings, into `seeds_lrp/keras_seed<s>/` | how much the maps (and the blocked classes) depend on the training seed; seed 42 redoes 3544560 with the new normalization | - |
| 3582374-76 | gpu-a100, 4 h | the same, torch | the same for Torch; seed 42 redoes 3544561 | - |
| 3582377 | gpu-a100, 30 min | `seeds_lrp.slurm --compare`: Pearson per map seed vs seed per backend and Keras vs Torch per seed, metrics, all-zero counts; `seeds_lrp/compare/compare_maps.log`, `seeds_lrp.3582377.out` | | 3582371-76 any |
| 3582378 | gpu-a100, 2 h | worktree `test_forcing.slurm` (log in the worktree's `scripts_TomsQoI/results/checks/test_forcing.3582378.out`, run dir `forcing_3582378/`) | data copy; iwetC2d base (the check prints both bases, the loader prints the one used); variance re-anomalizing removes; SST identical to main (`same_as_main/compare_maps.log`: every map pearson 1, max 0); every script on `--dataset forcing`, both anomaly options | - |
| 3582379 | gpu-a100, 16 h | worktree `forcing_anomalies_full.slurm`: Torch, 9 lags, 500 epochs, as-is vs reanomalize; `fanom_3582379/` | does re-anomalizing change accuracy and maps? The as-is run is also the first full wind-stress -> AMOC result | 3582378 ok |

### 2026-11-16: Nov 11 jobs finished, forcing data copied, Torch full pipeline queued

- **The colleague's folder became readable (2026-11-16).** Copied all of `INPUT_FORCINGS/` (9 fields of 2.66 GB, 60646 wet points x
  10958 days: atmPload, oceFWflx, oceQnet, oceQsw, oceSPflx, oceSflux, oceTAUE, oceTAUN, sIceLoad; `iwetC2d`, `rac_wet.bin`) and
  `TARGET_QoIs/` (RAPID QoI) to `/work/07665/shrey911/ls6/LRP_eccov4r5_forcing_data/` (23 GB), every file `cmp`-identical to the
  source; their `GRID/hFacC.data` is identical to ours. 3582378 will find the copy and skip copying.
- **All 12 Nov 11 gpu-a100 jobs COMPLETED, every run exit 0** (3573742, the final map comparison, still pending). Checked that the
  long ones ran to the end (e.g. 3573734 Keras OI reached iteration 10000 at all lags in 425 s; negative_class all 9 lags).
  - `LRP_compare_saved_all` (3573731): published networks, all 9 lags, Keras vs Torch Pearson 1.0000 for every map (both methods,
    both classes), max probability difference 1.7-2.1e-6. 22 non-finite positive samples at one lag, identical in both libraries.
  - **Dead/blocked Keras networks, seed study (3573700):** whole-class blocked (`blocked_share` 1.000) in Keras graph+TF32 (as the
    LRP script) 3 of 15 networks (seed 42 lags 0 and 30 neg, seed 44 lag 0 neg), Keras eager + no TF32 2 of 9 (seed 44 lags -30
    and 0 neg), **Torch 0 of 15** (max 0.275, lag 30 neg). So it is Keras training, neither one seed nor graph mode/TF32. **Keras
    training is not reproducible**: seed 42 now blocks lags 0 and 30, 3544560 (same seed) blocked -30, 0 and 120; Torch seed 42
    reproduces 3544561's blocked shares exactly (lag 30 0.268, lag 60 0.186). Neurons are not dead in any of them.
  - **Keras OI speed: resolved.** `check_oi_profile2` (3582278): warm-up (1 step, cold node) 336 s, then 400 steps 30 s, 3000 steps
    31 s, 3000 with prints every 100 32 s. The slowness was cold-node startup (library loading, data read); the steps are cheap.
  - Rules test (3582364/65) passed; preliminary (lag 0, first 50 neg samples, not representative): LRP-Z, epsilon and
    gradient x input are all-zero on the same 38% of samples as A1B0-B, i.e. the logit is locally constant there, so no attribution
    method can explain those samples. The full jobs 3582366-69 decide.
- **Torch full pipeline (owner: all earlier results for Torch, OI and LRP)**: `results/checks/torch_full_pipeline.slurm` (not in git):
  data reader plots; `LRP_torch` (train 9 lags, 500 epochs, seed 42, new normalization); OI of those networks and of the published
  networks in Torch (notebook settings); analyze (qoi_pred, accuracy/F1 whole/train/test) for both; `results_viz_new` figures (per lag
  and per result, correlation) and `results_viz_qoi_pred`. First runs of `eccov4r5_dataReader.py` (plots), `eccov4r5_analyze_torch.py`,
  `eccov4r5_results_viz_new.py`, `results_viz_qoi_pred.py`. **3587997** (dev, test: lags 0 60, 3 epochs, 20 OI steps, run dir
  `results/checks/torch_pipeline_test/`) then **3587998** (gpu-a100, 4 h, `afterok`, run dir **`scripts_TomsQoI/results/torch_full_sst/`**;
  log `results/checks/torch_full_pipeline.<id>.out`, one line per step and the list of figures). If the test fails, 3587998 stays
  `DependencyNeverSatisfied`: fix, `scancel`, resubmit both (commands in the header of the slurm file).

**Standing task from the owner (2026-11-16): get the forcing dataset to its full run.** Whatever breaks in 3582378 (the forcing test)
or 3582379 (the full run): read the logs (`<worktree>/scripts_TomsQoI/results/checks/test_forcing.<id>.out`, the per-step `.log` in
`forcing_<id>/`), fix the code on `multi_field`, resubmit the test and the full run (`afterok`), and repeat until the full run has
finished, **with both kinds of anomalies** (`--forcing-anomalies as-is` and `reanomalize`, as `forcing_anomalies_full.slurm` does),
then report accuracy/F1 per lag and the map comparison of the two. Small tests may go to `gpu-a100-dev`, the full run to `gpu-a100`.

**Forcing jobs, 2026-11-16 (owner: "throw in as many jobs as needed, the wait times are long")**: no longer chained behind the test,
so they start as soon as nodes free up; if the code is broken they fail fast and the test logs say why. **3582379 was cancelled**
(replaced by 3588019-21, run in parallel). `load_forcing` now sets non-finite values to 0 and prints how many (as the colleague's
`nan_to_num`). New `<worktree>/scripts_TomsQoI/results/checks/forcing_pipeline.slurm` (not in git): Torch, 9 lags, 500 epochs:
LRP_torch, OI_torch (both classes, 10000 steps), analyze_torch, results_viz_qoi_pred; output `<worktree>/scripts_TomsQoI/results/forcing_<name>_<jobid>/`,
log `results/checks/forcing_pipeline.<jobid>.out` (worktree).

| job | what |
|---|---|
| 3582378 | forcing test, gpu-a100 (kept as is) |
| 3588018 | the same test on gpu-a100-dev (starts sooner); log `test_forcing.3588018.out`, run dir `forcing_3588018/` |
| 3588019 | `wind_asis`: oceTAUE oceTAUN, as-is, no scaling (**the main wind -> AMOC result**) |
| 3588020 | `wind_reanom`: the same, reanomalized |
| 3588021 | `--compare` of 3588019 and 3588020 (maps and metrics; `forcing_anomalies_<id>/`), afterany |
| 3588022 / 3588023 | oceTAUE only / oceTAUN only, as-is |
| 3588024 | `buoyancy`: oceQnet oceFWflx, as-is, `--field-scale std` (different units) |
| 3588025 | `all9`: all nine fields, as-is, std, 40 h limit (545k inputs, ~24 GB of X in memory; may run out of GPU/CPU memory: if so, fix the trainer's data handling or drop fields) |

If any fails: read its `.out` and the step `.log`, fix on `multi_field`, resubmit with the same arguments (in the table's names).

### 2026-11-18: results and fixes

- **iwetC2d is 1-based, confirmed** (forcing test 3588018, `check_data.log`): base 1: all 60646 positions wet, agreement with hFacC > 0
  100.000%, RAC at the positions = `rac_wet.bin` to 5.9e-8 for every point; base 0 (the colleague's script): 1378 on dry cells,
  agreement 97.383% (his ~97.4%), RAC matches 35.6%. So the colleague's maps are shifted by one cell. Our loader detected base 1.
- Forcing tests 3582378 and 3588018 FAILED only in `check_data` (NaN in `anomalize_new`): every other step exit 0, SST identical to
  main (`same_as_main`), every script runs on `--dataset forcing` with both anomaly options. oceTAUE and oceTAUN each have **2227
  columns NaN on every day** (24403466 values, 3.67%); the loader sets them to 0. `check_forcing_data.py` now drops all-NaN columns
  (and zeroes other NaN); rerun alone: **3591817** (dev, `check_forcing_data.slurm`, log `check_forcing_data.3591817.out`), gives
  the re-anomalizing variance numbers.
- **The unscaled forcing runs did not learn**: 3588022 (tauE) and 3588023 (tauN) completed, but accuracy ~0.52 and F1 0 at every
  lag (one class predicted for all samples); val loss 0.693 = ln 2 throughout training, also in 3588019 (wind, running). Diagnosis:
  wind-stress anomalies have std 0.076 N/m^2, and the first layer's L2 penalty of 10 (tuned for SST in degC) forbids the large
  weights they would need. **Cancelled 3588019, 3588020, 3588021**; resubmitted with `--field-scale std`: **3591820** `wind_asis_std`,
  **3591821** `wind_reanom_std`, **3591822** compare (afterany), **3591823** `tauE_std`, **3591824** `tauN_std`. 3588024 (buoyancy)
  and 3588025 (all9) already use std, kept. Quick check that std learns: **3591819** (dev, `check_forcing_learns.slurm`, lag 0,
  100 epochs, wind and tauE; log `check_forcing_learns.3591819.out`): val loss must fall clearly below 0.693. If it does not, the
  problem is deeper (L2 of 10 on 60k-121k inputs, learning rate, or little skill), and the std runs will also fail: then try a
  smaller first-layer L2 (make_layers) and say so to the owner.
- **Rules on all samples (3582366-68 done; summary 3582369 pending)**, published networks, all 9 lags:
  - Bounds: map Pearson Bounded(-1,1) vs Bounded(training range) **0.94-0.96 positive, 0.86-0.90 negative**; vs no input rule
    0.92-0.95 / 0.91-0.96. Over all samples the bounds matter less than the first-400 check said (0.78-0.90), but not nothing.
  - Negative class: A1B0-B all-zero 0.17-0.27 at lags 0-90 (0.06 at -60, ~0 elsewhere). **LRP-Z, epsilon and gradient x input
    (identical, as expected for these networks) leave only 0.00-0.05 all-zero**: 81-95% of the A1B0-B-zero samples get relevance
    from them. So most blocked samples are not locally constant: A1B0 discards their (negative) contributions. The 38% of the
    50-sample test was not representative. A2B1-IB gives the same zero shares as Z. Signed rules: 32-49% of the relevance mass
    negative. Mean-map correlation with the class composite: A1B0-B -0.19 to -0.39 (neg), +0.21 to +0.46 (pos); Z similar.
- **Seed runs (3582371-76 done; compare 3582377 pending)**: classes with all-zero relevance in every sample ("i.e. 100.0 %" lines):
  Keras seed 42: 8 lines, 43: 2, 44: 2; Torch: 0 for every seed. Accuracy per lag: Keras 0.906-0.943, Torch 0.906-0.946, every seed.
- **Hedge against the L2 diagnosis being incomplete (owner: "submit as many jobs as needed to finish asap, I can't check every
  day")**: new option `--l2-first` (`eccov4r5_common.py`, default 10.0 = published SST; `make_layers(n_features, l2_first)`), and
  `forcing_pipeline.slurm` passes `EXTRA` from the environment (`sbatch --export=ALL,EXTRA="--l2-first 0.1" ...`). All with
  `--field-scale std`, Torch, full settings; run dirs `<worktree>/scripts_TomsQoI/results/forcing_<name>_<jobid>/`:

  | L2 first layer | wind as-is | wind reanomalized | compare | tauE | tauN | buoyancy (Qnet+FWflx) | all 9 |
  |---|---|---|---|---|---|---|---|
  | 10 | 3591820 | 3591821 | 3591822 | 3591823 | 3591824 | 3588024 | 3588025 |
  | 1 | 3591828 | 3591829 | 3591830 | 3591837 | 3591838 | 3591839 | |
  | 0.1 | 3591831 | 3591832 | 3591833 | 3591840 | 3591841 | 3591842 | 3591843 |
  | 0.01 | 3591834 | 3591835 | 3591836 | | | | |

  **3591844** (`forcing_summary.slurm`, afterany of all of them): one table of accuracy/F1 per lag for every forcing run, with
  "learned: True/False"; log `<worktree>/scripts_TomsQoI/results/checks/forcing_summary.3591844.out`. **Read that first.** Pick
  the L2 that learns best (highest test-period accuracy without collapse); its as-is vs reanomalized compare answers the anomaly
  question. If none learns at any L2: the problem is not the penalty (check the QoI alignment, the class balance, the learning rate).
- **Torch SST pipeline test 3587997 passed**: every step exit 0, 27 figures (first runs of the data reader plots,
  `eccov4r5_analyze_torch.py`, `eccov4r5_results_viz_new.py`, `results_viz_qoi_pred.py`); full run 3587998 pending.

Notes: the seed jobs read the main checkout's scripts when they start; merging `multi_field` before then is harmless (the SST path is
meant to be identical, which 3582378 checks) but do it only after 3582378 passed. The dev queue is full (3 jobs per user) until 3582278
and the rules test finish. When back, read in this order: `rules_all_lags.3582369.out` (two open questions), `seeds_lrp/compare/compare_maps.log`,
the forcing test and `fanom_3582379`, then the Nov 11 jobs as listed above.
- **2026-11-21, read by a new session (the original long session, 3cf0f372, resumed): all jobs above finished except the four as-is vs
  reanomalized compares 3591822 / 3591830 / 3591833 / 3591836 (pending), all9 at L2 0.1 3591843 (running, 2.6 h at 01:18) and the
  summary 3591844 (afterany). Every finished one exit 0 except the two forcing tests (known, `check_data` only).**
  - **The forcing networks do not generalize.** The 0.89-0.94 "accuracy" of `metrics_torch.json` / `run_experiment` is over *all*
    samples, training included. `analyze_torch.log` of each run splits it: **train 94-100%, test 43-63%** at every lag for wind (as-is),
    tauE, tauN, buoyancy and all9, at every L2 (10, 1, 0.1, 0.01). The test-period majority class is about 55% (the unscaled tauE/tauN
    runs, which predict one class, score 54-57% on test), so these are at or below a constant predictor. Best: **wind reanomalized,
    lag 0: test 67-71%** (L2 10 std 71, L2 1 69, L2 0.1 67, L2 0.01 68); other reanomalized lags 48-59. Per run, test accuracy per
    lag -60..180 (%):
    wind as-is std (L2 10) 53 56 63 51 47 46 48 48 52; wind reanom std 59 57 71 57 52 53 54 48 53; tauE std 55 56 62 51 49 48 47 47 52;
    tauN std 51 56 60 50 47 50 53 46 54; buoyancy 46 52 48 53 50 42 50 49 49; all9 54 59 54 47 46 44 48 46 46.
    So "learned" in the summary job 3591844 (which reads the all-sample accuracy) will say True for runs that only memorised.
  - **The published SST setup overfits too**, less: Torch SST (`results/torch_full_sst/analyze_torch.log`, 500 epochs, seed 42):
    train 99.0-99.5%, test 61 58 67 67 66 54 59 73 68 (lags -60..180); the published networks in Torch
    (`analyze_torch_published.log`): train 99.4-99.6%, test 62 56 67 67 67 58 62 72 68. Above the ~55% baseline at most lags, but far
    from the ~0.93 all-sample accuracy quoted so far.
  - **Likely reason the checkpoint does not stop it**: the validation set is a *shuffled random* 20% of the training period
    (`split_lag`, `train_test_split(shuffle=True)`), and daily anomalies (and the 61-day running-mean QoI) are strongly autocorrelated,
    so every validation day has near-copies in training: val loss measures interpolation, not generalization, and keeps improving to
    epoch 500. A blocked (contiguous) validation period, or a gap between train/val/test, is the standard fix. Not changed: it alters
    the published experiment, the owner decides. Also an option for forcing: far fewer inputs (60k-121k inputs for ~8k samples).
- **2026-11-21, owner: "Try both runs with a contiguous separate validation, and also the current way. I want to see the LRP plots
  for both, so 2 sets of plots for SST and forcing each; within forcing both normal and re-anomalization" and "do whatever is needed
  to prevent overfit and still get decent results".** Done on `multi_field` (worktree), commit "Contiguous validation, input
  coarsening, train/val/test metrics":
  - `--val-split shuffled|contiguous` (default `shuffled` = the published experiment) and `--val-gap` (default 60 days):
    `split_lag(..., mode, gap)`; contiguous = train | gap | last 20% of the training period as validation | gap | test, so no
    validation day has a neighbour in training or test; `split_for(ctx, lag)` used by `run_experiment`, `train_on_lag`,
    `eccov4r5_dead_neurons.py`.
  - `--coarsen k` (default 1 = none): `coarse_index` groups the wet points by k x k blocks of each tile (wet cells only),
    `coarsen_fields` averages X per field (sparse matrix) in `make_context` (`ctx.coarse = (index, m)`; prints
    "coarsened kxk: n -> m inputs"); `field_maps(..., coarse)` draws every map (LRP, OI, composites, compare) back on the full grid,
    each wet cell at its block's value. Against overfitting: k^2 fewer inputs (SST 31811 -> ~3.5k at k 3, ~0.9k at k 6).
  - `run_experiment` metrics now also have `Train_/Val_/Test_Accuracy`, `_F1` and `Test_Majority` (the test-period majority-class
    share, the baseline to beat), printed per lag as "lag L: accuracy train .., val .., test .. (majority ..)". Defaults leave the
    published path unchanged (only extra metric keys; `f1_score(zero_division=0)` in the new ones).
  - Runs (Torch, 500 epochs, seed 42, all 9 lags, LRP + OI + analyze per run), via the new one-off
    `<worktree>/scripts_TomsQoI/results/checks/gen_pipeline.slurm <name> [test] -- <options>` (output
    `<worktree>/scripts_TomsQoI/results/gen_<name>_<jobid>/`, log `results/checks/gen_pipeline.<jobid>.out`):
    tests **3601307** (SST) and **3601308** (forcing reanomalized), dev, lags 0 60, 3 epochs, contiguous, L2 1, coarsen 3; then the
    grid, all `--val-split contiguous`, **afterok of both tests**: L2 {10, 1, 0.1} x coarsen {1, 3, 6} for SST (3601309-3601317), forcing
    wind (oceTAUE oceTAUN, std) as-is (3601318-3601326) and reanomalized (3601327-3601334, and L2 0.1 coarsen 6 submitted later, see
    below). Order within each dataset: (10,1) (10,3) (10,6) (1,1) (1,3) (1,6) (0.1,1) (0.1,3) (0.1,6). IDs in
    `results/checks/gen_grid_ids.txt`.
  - **Submit limit: 34 jobs per user** (`QOSMaxSubmitJobPerUserLimit`). The last grid run (reanom L2 0.1 coarsen 6) and the summary
    did not fit; a **detached** watcher, `<worktree>/scripts_TomsQoI/results/checks/submit_rest.sh` (started with `setsid nohup`, PID 3911624 on the login node, log `submit_rest.log` next to it), submits them when slots free up (after the two dev tests), and survives the session closing. **If it did not** (or the login node was rebooted)
    (no `gen_reanom_contig_l2_0.1_c6_*` dir / no `gen_summary.*.out`), from the worktree:
    `sbatch -t 10:00:00 scripts_TomsQoI/results/checks/gen_pipeline.slurm reanom_contig_l2_0.1_c6 -- --dataset forcing --fields oceTAUE oceTAUN --field-scale std --forcing-anomalies reanomalize --val-split contiguous --l2-first 0.1 --coarsen 6`
    then `sbatch --dependency=afterany:<all grid ids> scripts_TomsQoI/results/checks/gen_summary.slurm`.
    **Done (01:36)**: the watcher submitted **3601338** (reanom L2 0.1 coarsen 6, no dependency) and the summary **3601339**
    (afterany of 3601309-3601334 and 3601338; log `<worktree>/scripts_TomsQoI/results/checks/gen_summary.3601339.out`).
  - **Interim results (2026-11-23 18:56)**: tests 3601307/08 passed; done: all 9 SST runs, as-is (L2 10 c1/c3/c6, L2 1 c1),
    reanom L2 0.1 c6 (3601338); 13 forcing runs and the summary 3601339 pending. Means over the 9 lags (from the "lag L: accuracy
    train/val/test (majority)" lines of each `gen_*/LRP_torch.log`):

    | run | train | val | test | test majority |
    |---|---|---|---|---|
    | SST current way (shuffled, `torch_full_sst`) | ~0.99 | - | 0.637 (0.54-0.73) | ~0.555 |
    | SST L2 10 c1 | 0.898 | **0.799** | 0.656 | 0.555 |
    | SST L2 10 c3 | 0.871 | 0.794 | **0.676** (0.62-0.72) | 0.555 |
    | SST L2 10 c6 | 0.812 | 0.766 | 0.681 | 0.555 |
    | SST L2 1 c1 / c3 / c6 | 0.95 | 0.76-0.78 | 0.641 / 0.628 / 0.648 | 0.555 |
    | SST L2 0.1 c1 / c3 / c6 | 0.98-0.99 | 0.73-0.76 | 0.624 / 0.633 / 0.628 | 0.555 |
    | forcing as-is L2 10 c1 / c3 / c6 | 0.61-0.62 | 0.30-0.37 | 0.535 / 0.537 / 0.553 | 0.554 |
    | forcing as-is L2 1 c1 | 0.685 | 0.492 | 0.518 | 0.554 |
    | forcing reanom L2 0.1 c6 | 0.859 | 0.560 | 0.544 | **0.729** |

    SST: contiguous validation stops the memorization (train 0.81-0.90 at L2 10) and raises test accuracy by 2-4 points with less
    spread over lags; the strong first-layer L2 of the published setup is the best; coarsening helps a little. Selection on
    validation picks L2 10 c1 (test 0.656); c3 is a near tie on validation and better on test. Forcing: at or below the majority
    baseline on test; validation accuracy below 0.5 for as-is (the contiguous validation block, the last ~5 years of the training
    period, is probably dominated by one AMOC class: the as-is anomalies are not detrended and the AMOC has multi-year regimes; the
    val majority share is not recorded, check that); reanomalizing changes the test class balance (majority 0.73) and the network
    is below it. No evidence so far that wind stress predicts the AMOC class with this network.
    **Context from the owner (2026-11-23): the LRP paper reported only 60-70% accuracy.** So the SST test accuracies (current way
    0.637, contiguous L2 10 0.66-0.68) are in line with the published result; the "overfitting" alarm of 2026-11-21 concerned the
    ~0.93 all-sample accuracy that these notes and job summaries had been quoting, not the paper. Contiguous validation is a modest
    gain at the top of the paper's range (worst lag ~0.62 instead of 0.54) and gives networks that do not memorize. Forcing (~0.53,
    at the majority baseline) is below the paper's SST level.
    **Owner's framing for the paper (2026-11-23):** the goal is whether LRP relevance matches the MITgcm adjoint sensitivities, not
    the best classifier, so the overfit (shuffled-validation) networks are acceptable; the caveat (LRP explains the network, and an
    overfit network's relevance may partly reflect memorized training-set patterns, so a mismatch with the adjoint is hard to
    interpret) **will be discussed in the paper**. Useful evidence for that discussion: the correlation of the shuffled vs contiguous
    SST LRP maps (offered: `eccov4r5_compare_maps.py` on `torch_full_sst/LRP_torch` vs the best `gen_sst_contig_*` run, not queued
    yet), optionally LRP on test-period samples only. Forcing networks at chance should not be compared with the adjoint.
  - **Forcing plots looked at (2026-11-23)**, copies for the owner in `scripts_TomsQoI/results/forcing_plots/` (main checkout,
    gitignored): `1_shuffled_asis` (3591820), `2_shuffled_reanom` (3591821), `3_contiguous_asis_L2-10` (3601318), each
    `LRP_<field>_<cls>.png` (LRP-A1B0-B) and `OI_<field>_<cls>.png`, `SOURCE.txt` names the run. Observations: shuffled as-is
    **lag 0 LRP blank**: 100% of both classes all-zero (5454 pos, 4549 neg; A1B0 blocking, now in a Torch forcing network);
    shuffled reanom lag -60 pos 100% blocked; elsewhere 0-8%. LRP maps otherwise diffuse, weakly raised in the subpolar North
    Atlantic at lags -30..90. **OI: shuffled = mostly small-scale noise; contiguous L2 10 = smooth, large-scale, a strong North
    Atlantic dipole (+ tauE over the subpolar gyre / Labrador Sea, - over the subtropical gyre / Gulf Stream, NAO-like) at lags
    -30, 0, 30, fading by 60-90**, plus Southern Ocean bands. Lag 0 is also the only lag where forcing test accuracy was above chance
    (0.62-0.63 shuffled), consistent with the instantaneous Ekman response of the AMOC at 26N to the local wind.
  - **Four ideas to make forcing work (owner 2026-11-23: "Do all 4, schedule all the runs")**, code on `multi_field` (commit
    "Forcing: high-passed QoI classes, time-mean inputs, regions"), all forcing wind (oceTAUE oceTAUN), std, contiguous validation, L2 10:
    1. **High-passed QoI classes** instead of regression (regression would rework the output layer, loss, LRP sample selection and OI
       of every script; told the owner): `--qoi-highpass N` removes the centered N-day running mean from the daily QoI before the
       61-day mean and sign (`load_data_with_y`, `running_mean`), so the classes follow AMOC variability faster than N days, not its
       multi-year regimes.
    2. **Time-integrated inputs**: `--input-mean N`, trailing N-day mean of each input (fewer days at the start), in column blocks.
    3. **Longer lags**: `--lags` was already generic; `gen_pipeline.slurm` now takes `LAGS` from the environment
       (`sbatch --export=ALL,LAGS="..."`). Long set: 0 90 180 365 545 730 1095 1460 1825 days.
    4. **Fewer inputs**: `--region natl` (`REGIONS`: 0-80N, 100W-20E) keeps the North Atlantic wet points (in `make_context`, before
       `--coarsen`); maps NaN outside; prints "region natl: m of n wet points kept per field".
    Jobs: test **3609296** (dev; all four at once: hp 365, mean 90, natl, coarsen 3, lags 0 1095, 3 epochs); then, `afterok`, on
    gpu-a100 (10 h): 3609297 `f_hp365_asis` (coarsen 3), 3609298 `f_hp365_reanom`, 3609299 `f_mean30`, 3609300 `f_mean90`
    (coarsen 3), 3609301 `f_longlags` (coarsen 3, long lags), 3609302 `f_natl_c1`, 3609303 `f_natl_c3`, 3609304 `f_combo_hp`
    (natl + mean 90 + hp 365, coarsen 3), 3609305 `f_combo_long` (natl + mean 90, long lags, coarsen 3); run dirs
    `<worktree>/scripts_TomsQoI/results/gen_f_<name>_<id>/`. Summary **3609306** (`forcing4_summary.py/.slurm`, afterany):
    `<worktree>/scripts_TomsQoI/results/checks/forcing4_summary.3609306.out`: per run train/val/test/majority per lag and the LRP and
    OI figure paths. Note: with `--qoi-highpass` the class balance changes, compare test accuracy with that run's own majority share.
    IDs in `results/checks/forcing4_ids.txt`.
  - **3609284** (dev, `<worktree>/.../checks/compare_valsplit.slurm`): Pearson per map and lag, SST shuffled (`torch_full_sst`) vs
    contiguous L2 10 c1 and c3 (LRP and OI), forcing as-is shuffled vs contiguous L2 10 c1, forcing shuffled as-is vs reanom; log
    `<worktree>/scripts_TomsQoI/results/checks/compare_valsplit.3609284.out`.
  - **Cancelled with the owner's approval** ("kill whatever is unnecessary"): 3591830, 3591833, 3591836 (as-is vs reanomalized map
    compares at L2 1, 0.1, 0.01 of overfit networks) and 3591844 (forcing summary judging "learned" on all-sample accuracy). Kept:
    3591822 (the L2 10 compare, matching the current-way plots) and 3591843 (all9 at L2 0.1, running).
  - **Summary**: `results/checks/gen_summary.py` / `.slurm` (worktree): per run and lag train/val/test accuracy and the majority
    baseline; per dataset the configuration with the **best mean validation accuracy** (chosen on validation, never on test), its
    test accuracy and its LRP figures; and the current-way runs' test accuracies and figure folders. **The two sets of LRP plots the
    owner asked for**: current way = `results/torch_full_sst/LRP_torch/*.png` (main checkout; SST),
    `<worktree>/scripts_TomsQoI/results/forcing_wind_asis_std_3591820/LRP_torch/*.png` and `forcing_wind_reanom_std_3591821/LRP_torch/*.png`
    (forcing, L2 10, std, shuffled); contiguous = the best run per dataset in `gen_summary.<id>.out`.
  - If the tests fail, the grid stays `DependencyNeverSatisfied`: read `results/checks/gen_pipeline.360130{7,8}.out` and the step logs,
    fix, `scancel` the grid, resubmit (loop in this session's transcript).
- A collaborator-facing summary of all conclusions was published (private until the owner shares it):
  https://claude.ai/artifact/RngeEWQRRXVYxDvQsDX5gT (source in that session's scratchpad; republish from a new session by passing the URL).

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
