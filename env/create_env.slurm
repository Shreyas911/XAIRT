#!/bin/bash
#SBATCH -J create_XAIRT_env
#SBATCH -A Transfer-of-ECCO-glo
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -o create_XAIRT_env.%j.out
#SBATCH -e create_XAIRT_env.%j.err

# Builds the XAIRT environment on a GPU node. The login node caps the virtual memory of a process
# at 8 GB, which makes mamba fail with std::bad_alloc while it extracts packages.
#
#   sbatch env/create_env.slurm          (from the repository root)

MAMBA_DIR=/work/07665/shrey911/ls6/mambaforge
export PATH=$MAMBA_DIR/bin:$PATH

# The home directory has a 10 GB quota that pip's cache (torch wheels) fills up, which fails the pip step
# with "Disk quota exceeded". Keep the cache on /work.
export PIP_CACHE_DIR=/work/07665/shrey911/ls6/pip_cache

cd "$SLURM_SUBMIT_DIR"

echo "Node: $(hostname), ulimit -v: $(ulimit -v)"
nvidia-smi -L

# A GPU node has a driver, so no CONDA_OVERRIDE_CUDA
mamba env create -f env/XAIRT.yaml || exit 1

echo
mamba run -n XAIRT pip list 2>/dev/null | grep -i -E "^(numpy|protobuf|keras|tensorflow|torch|captum|innvestigate) "
mamba run -n XAIRT python -c "import tensorflow as tf, torch; print('TF GPUs:', tf.config.list_physical_devices('GPU'), 'torch cuda:', torch.cuda.is_available())"
