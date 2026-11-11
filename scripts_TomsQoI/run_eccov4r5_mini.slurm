#!/bin/bash
#SBATCH -J XAIRT_mini
#SBATCH -A Transfer-of-ECCO-glo
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -o scripts_TomsQoI/results/logs/XAIRT_mini.%j.out
#SBATCH -e scripts_TomsQoI/results/logs/XAIRT_mini.%j.err

# A miniature of every run of run_eccov4r5.slurm, the same commands with small settings: lags 0 and 60, 3 epochs, 20 OI
# steps, 20 LRP samples per class. Only checks that each script runs to completion, before the long runs are queued. The
# *_trained runs use the networks this job trains, as the real jobs use those of an LRP_keras / LRP_torch run. From the
# repository root:
#
#   sbatch scripts_TomsQoI/run_eccov4r5_mini.slurm
#
# A failing script does not stop the others; its exit code and the end of its log are printed to the job log.

# Not conda activate, which does not put the env first on PATH in a batch job. Without the env's lib first on
# LD_LIBRARY_PATH, TensorFlow finds the wrong libcublas.so.11 (the cuda/11.4 module) and sees no GPU.
ENV=/work/07665/shrey911/ls6/mambaforge/envs/XAIRT
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH

cd "$SLURM_SUBMIT_DIR"
RUN_DIR=$PWD/scripts_TomsQoI/results/mini_$SLURM_JOB_ID
mkdir -p "$RUN_DIR"
S=$PWD/scripts_TomsQoI
MODELS=$PWD/examples_TomsQoI/LRP_output_forHelen/saved_models
echo "Node: $(hostname), python: $(which python), run dir: $RUN_DIR"

run() {
    local name=$1; shift
    local t0=$SECONDS
    ( cd "$RUN_DIR" && python -u "$@" ) > "$RUN_DIR/$name.log" 2>&1
    local rc=$?
    echo "$name: exit $rc ($((SECONDS - t0)) s)"
    [ $rc -ne 0 ] && tail -5 "$RUN_DIR/$name.log" | sed 's/^/    /'
}

LAGS="--lags 0 60"
OI="$LAGS --kind both --oi-iters 20 --print-freq 10"
run OI_keras       $S/eccov4r5_OI_keras.py --source saved       $OI --saved-models-dir $MODELS --out-dir OI_keras
run OI_torch       $S/eccov4r5_OI_torch.py --source saved-keras $OI --saved-models-dir $MODELS --out-dir OI_torch
run LRP_compare    $S/eccov4r5_LRP_A1B0_compare.py --source saved --saved-models-dir $MODELS --lags 0 --out-dir LRP_compare
run negative_class $S/eccov4r5_negative_class_LRP.py --source saved --saved-models-dir $MODELS $LAGS --n-lrp 20 --out-dir negative_class
run LRP_keras      $S/eccov4r5_LRP_A1B0_keras.py $LAGS --epochs 3 --out-dir LRP_keras
run LRP_torch      $S/eccov4r5_LRP_A1B0_torch.py $LAGS --epochs 3 --out-dir LRP_torch
KERAS_MODELS=$RUN_DIR/LRP_keras/models
TORCH_MODELS=$RUN_DIR/LRP_torch/models
run LRP_compare_trained       $S/eccov4r5_LRP_A1B0_compare.py --source saved --saved-models-dir $KERAS_MODELS $LAGS --out-dir LRP_compare_trained
run OI_keras_trained          $S/eccov4r5_OI_keras.py --source saved       $OI --saved-models-dir $KERAS_MODELS --out-dir OI_keras_trained
run OI_keras_in_torch_trained $S/eccov4r5_OI_torch.py --source saved-keras $OI --saved-models-dir $KERAS_MODELS --out-dir OI_keras_in_torch_trained
run OI_torch_trained          $S/eccov4r5_OI_torch.py --source saved-torch $OI --saved-models-dir $TORCH_MODELS --out-dir OI_torch_trained
run LRP_compare_torch_trained $S/eccov4r5_LRP_A1B0_compare.py --source saved-torch --saved-models-dir $TORCH_MODELS $LAGS --out-dir LRP_compare_torch_trained
run OI_torch_in_keras_trained $S/eccov4r5_OI_keras.py --source saved-torch $OI --saved-models-dir $TORCH_MODELS --out-dir OI_torch_in_keras_trained
echo "done"
