#!/bin/bash
#SBATCH -J XAIRT_ecco
#SBATCH -A Transfer-of-ECCO-glo
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o scripts_TomsQoI/results/logs/XAIRT_ecco.%j.out
#SBATCH -e scripts_TomsQoI/results/logs/XAIRT_ecco.%j.err

# Runs the ECCOv4r5 scripts of scripts_TomsQoI, the ones named on the command line. Each logs to its own file in $RUN_DIR;
# a failing script does not stop the others. From the repository root:
#
#   sbatch scripts_TomsQoI/run_eccov4r5.slurm OI_keras OI_torch
#
# The settings are small, for a first run: lag 0, 1000 OI steps. The networks are the saved Keras models (Torch copies them).

# Not conda activate, which does not put the env first on PATH in a batch job. Without the env's lib first on
# LD_LIBRARY_PATH, TensorFlow finds the wrong libcublas.so.11 (the cuda/11.4 module) and sees no GPU.
ENV=/work/07665/shrey911/ls6/mambaforge/envs/XAIRT
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH

cd "$SLURM_SUBMIT_DIR"
RUN_DIR=$PWD/scripts_TomsQoI/results/ecco_$SLURM_JOB_ID
mkdir -p "$RUN_DIR"
S=$PWD/scripts_TomsQoI
MODELS=$PWD/examples_TomsQoI/LRP_output_forHelen/saved_models
echo "Node: $(hostname), python: $(which python), run dir: $RUN_DIR"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1

SELECTED="$*"
[ -z "$SELECTED" ] && { echo "Name the runs, e.g. sbatch scripts_TomsQoI/run_eccov4r5.slurm OI_keras OI_torch"; exit 1; }
run() {
    local name=$1; shift
    [[ " $SELECTED " != *" $name "* ]] && return
    local t0=$SECONDS
    ( cd "$RUN_DIR" && python -u "$@" ) > "$RUN_DIR/$name.log" 2>&1
    echo "$name: exit $? ($((SECONDS - t0)) s)"
}

OI="--lags 0 --kind both --oi-iters 1000 --print-freq 100 --saved-models-dir $MODELS"
run OI_keras $S/eccov4r5_OI_keras.py --source saved       $OI --out-dir OI_keras
run OI_torch $S/eccov4r5_OI_torch.py --source saved-keras $OI --out-dir OI_torch

# Keras vs Torch on the same saved network, LRP-A1B0-B (Torch's BoundedRule) and LRP-A1B0, lag 0
run LRP_compare  $S/eccov4r5_LRP_A1B0_compare.py --source saved --saved-models-dir $MODELS --lags 0 --out-dir LRP_compare
# The all-zero relevance of the negative class, all 9 lags (HANDOFF.md, "Open research question")
run negative_class $S/eccov4r5_negative_class_LRP.py --source saved --saved-models-dir $MODELS --n-lrp 300 --out-dir negative_class

# Full runs, as the notebook: train for every lag (500 epochs) and explain. Long, give sbatch a longer -t.
run LRP_keras $S/eccov4r5_LRP_A1B0_keras.py --out-dir LRP_keras
run LRP_torch $S/eccov4r5_LRP_A1B0_torch.py --out-dir LRP_torch
# The networks of an LRP_keras run explained by both libraries. KERAS_MODELS is the models/ of that run, e.g.
#   sbatch -t 06:00:00 --dependency=afterok:<id> --export=ALL,KERAS_MODELS=$PWD/scripts_TomsQoI/results/ecco_<id>/LRP_keras/models \
#          scripts_TomsQoI/run_eccov4r5.slurm LRP_compare_trained
#
# OI with the notebook's settings (all lags, both kinds, 10000 steps) on the networks of an LRP_keras run (KERAS_MODELS) and of an
# LRP_torch run (TORCH_MODELS, the .pt files): Keras, the Keras networks in Torch, and Torch.
needs() {    # needs <run> <variable>: true if the run was named and the variable is empty
    [[ " $SELECTED " == *" $1 "* && -z "${!2}" ]] && { echo "$1 needs $2, the models/ directory of an LRP run"; return 0; }
    return 1
}
needs LRP_compare_trained KERAS_MODELS || run LRP_compare_trained $S/eccov4r5_LRP_A1B0_compare.py --source saved --saved-models-dir "$KERAS_MODELS" --out-dir LRP_compare_trained
needs OI_keras_trained KERAS_MODELS || run OI_keras_trained $S/eccov4r5_OI_keras.py --source saved       --saved-models-dir "$KERAS_MODELS" --out-dir OI_keras_trained
needs OI_keras_in_torch_trained KERAS_MODELS || run OI_keras_in_torch_trained $S/eccov4r5_OI_torch.py --source saved-keras --saved-models-dir "$KERAS_MODELS" --out-dir OI_keras_in_torch_trained
needs OI_torch_trained TORCH_MODELS || run OI_torch_trained $S/eccov4r5_OI_torch.py --source saved-torch --saved-models-dir "$TORCH_MODELS" --out-dir OI_torch_trained
# The reverse: the networks of an LRP_torch run (TORCH_MODELS) copied into Keras (torch_to_keras), explained by both libraries
# and with the Keras OI
needs LRP_compare_torch_trained TORCH_MODELS || run LRP_compare_torch_trained $S/eccov4r5_LRP_A1B0_compare.py --source saved-torch --saved-models-dir "$TORCH_MODELS" --out-dir LRP_compare_torch_trained
# Dead neurons (eccov4r5_dead_neurons.py): are the networks without relevance / OI movement for a whole class a property of Keras'
# training or of the seed? Train both backends for several seeds (seed 42 is the one of the LRP runs, so it should reproduce them),
# and Keras in eager mode without TF32, the two ways in which its training differs from the LRP script's. About 4 h, -t 08:00:00.
run dead_neurons_seeds $S/eccov4r5_dead_neurons.py --backends keras torch --seeds 42 43 44 --lags -30 0 30 60 120 --out-dir dead_neurons_seeds
run dead_neurons_keras_eager $S/eccov4r5_dead_neurons.py --backends keras --keras-eager --no-tf32 --seeds 42 43 44 --lags -30 0 120 --out-dir dead_neurons_keras_eager
needs OI_torch_in_keras_trained TORCH_MODELS || run OI_torch_in_keras_trained $S/eccov4r5_OI_keras.py --source saved-torch --saved-models-dir "$TORCH_MODELS" --out-dir OI_torch_in_keras_trained
echo "done"
