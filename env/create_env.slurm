#!/bin/bash
#SBATCH -J create_XAIRT_env
#SBATCH -A Transfer-of-ECCO-glo
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:20:00
#SBATCH -o create_XAIRT_env.%j.out
#SBATCH -e create_XAIRT_env.%j.err

# Builds the XAIRT environment on a GPU node, and checks it. The login node caps the virtual memory of a process
# at 8 GB, which makes mamba fail with std::bad_alloc while it extracts packages. If the environment is already
# there, it is not built again, only checked, which takes about a minute.
#
#   sbatch env/create_env.slurm          (from the repository root)
#
# A build took about 7 minutes (conda 3, pip 4). The time limit is short on purpose, a short job starts sooner.

MAMBA_DIR=/work/07665/shrey911/ls6/mambaforge
export PATH=$MAMBA_DIR/bin:$PATH

# The home directory has a 10 GB quota that pip's cache (torch wheels) fills up, which fails the pip step
# with "Disk quota exceeded". Keep the cache on /work.
export PIP_CACHE_DIR=/work/07665/shrey911/ls6/pip_cache

cd "$SLURM_SUBMIT_DIR"

echo "Node: $(hostname), ulimit -v: $(ulimit -v)"
nvidia-smi -L
nvidia-smi --query-gpu=driver_version --format=csv,noheader | sort -u | sed 's/^/Driver: /'

# A GPU node has a driver, so no CONDA_OVERRIDE_CUDA
if [ -d "$MAMBA_DIR/envs/XAIRT" ]; then
    echo "The XAIRT environment is there, only checking it (mamba env remove -n XAIRT to build it again)"
else
    mamba env create -f env/XAIRT.yaml || exit 1
fi

echo
echo "--- packages"
mamba run -n XAIRT pip list 2>/dev/null | grep -i -E "^(numpy|protobuf|keras|tensorflow|torch|captum|innvestigate|nvidia-nccl) "

# The order tensorflow, then torch is the one that the scripts use, and the one that a wrong NCCL breaks
echo "--- tensorflow then torch, and the GPUs"
mamba run -n XAIRT python -c "import tensorflow as tf, torch; print('TF GPUs:', tf.config.list_physical_devices('GPU'), 'torch cuda:', torch.cuda.is_available(), 'torch', torch.__version__, torch.version.cuda)"

echo "--- import XAIRT"
PYTHONPATH=src mamba run -n XAIRT python -c "import XAIRT; print('XAIRT ok')"
