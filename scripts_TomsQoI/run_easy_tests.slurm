#!/bin/bash
#SBATCH -J XAIRT_easy
#SBATCH -A Transfer-of-ECCO-glo
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:45:00
#SBATCH -o scripts_TomsQoI/results/logs/XAIRT_easy.%j.out
#SBATCH -e scripts_TomsQoI/results/logs/XAIRT_easy.%j.err

# Checks the XAIRT env on a GPU node, then runs the fake-data scripts of scripts_TomsQoI with small settings.
# Each script logs to its own file in $RUN_DIR; a failing script does not stop the others.
#
#   sbatch scripts_TomsQoI/run_easy_tests.slurm                       (from the repository root, all tests)
#   sbatch scripts_TomsQoI/run_easy_tests.slurm fake_pos fake_signed  (only these)

# Not conda activate, which does not put the env first on PATH in a batch job. Without the env's lib first on
# LD_LIBRARY_PATH, TensorFlow finds the wrong libcublas.so.11 (the cuda/11.4 module) and sees no GPU.
ENV=/work/07665/shrey911/ls6/mambaforge/envs/XAIRT
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH

cd "$SLURM_SUBMIT_DIR"
RUN_DIR=$PWD/scripts_TomsQoI/results/easy_$SLURM_JOB_ID
mkdir -p "$RUN_DIR"
S=$PWD/scripts_TomsQoI
echo "Node: $(hostname), python: $(which python), run dir: $RUN_DIR"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

echo "--- env"
pip list 2>/dev/null | grep -i -E "^(numpy|protobuf|keras|tensorflow|torch|captum|innvestigate) "
python -c "import tensorflow as tf, torch; print('TF GPUs:', len(tf.config.list_physical_devices('GPU'))); print('torch cuda:', torch.cuda.is_available(), torch.__version__, torch.version.cuda)"
PYTHONPATH=src python -c "import XAIRT; print('XAIRT ok')"

SELECTED="$*"
run() {
    local name=$1; shift
    [ -n "$SELECTED" ] && [[ " $SELECTED " != *" $name "* ]] && return
    local t0=$SECONDS
    ( cd "$RUN_DIR" && python -u "$@" ) > "$RUN_DIR/$name.log" 2>&1
    echo "$name: exit $? ($((SECONDS - t0)) s)"
}

echo "--- scripts"
run fake_pos      $S/fake_data_LRP_A1B0_compare.py --positive-inputs --epochs 20 --n-samples 1500 --n-explain 50 --out-dir fake_pos
run fake_signed   $S/fake_data_LRP_A1B0_compare.py                   --epochs 20 --n-samples 1500 --n-explain 50 --out-dir fake_signed
run LRP_keras     $S/simpleTests_LRP_keras.py     --epochs 2 --n-samples 500
run LRP_torch     $S/simpleTests_LRP_torch.py     --epochs 2 --n-samples 500
run MWE_keras     $S/LRP_manual_MWE_keras.py      --epochs 2 --n-samples 500
run MWE_torch     $S/LRP_manual_MWE_torch.py      --epochs 2 --n-samples 500
run underst_keras $S/understandingLRP_keras.py    --epochs 2 --n-samples 500
run underst_torch $S/understandingLRP_torch.py    --epochs 2 --n-samples 500
run OI_keras      $S/simpleTests_OI_keras.py      --epochs 1 --n-samples 20000
run OI_torch      $S/simpleTests_OI_torch.py      --epochs 1 --n-samples 20000
echo "done"
