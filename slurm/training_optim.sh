#!/bin/bash

# Instructing SLURM to locate and assign X number of nodes with Y number of cores in each node.
# X,Y are integers. Refer to table for various combinations. X will almost always be 1.
#SBATCH -N 1
#SBATCH -c 4

# Governs the run time limit and resource limit for the job. Pick values from the partition and QOS tables.
#SBATCH --gres=gpu:pascal:1
#SBATCH -p tpg-gpu-small
#SBATCH --qos=long-high-prio
#SBATCH -t 07-00:00:00
#SBATCH --mem=28G

#SBATCH --job-name=GWL_ray_tune_head
#SBATCH -o /home3/swlc12/msc-groundwater-gwl-parallel/slurm/logs/slurm-ray-%j.out

#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=swlc12@durham.ac.uk

# Faily immediately with errors, undefined vars or if any command fails (not just last)
set -euo pipefail

set +u; source /etc/profile; set -u
source /home3/swlc12/msc-groundwater-gwl/.venv_ncc/bin/activate

unset RAY_ADDRESS
unset RAY_HEAD_IP
ray stop --force || true

echo "Running on $(hostname)"
nvidia-smi || true

# Make code importable + set absolute roots
export PYTHONPATH="/home3/swlc12/msc-groundwater-gwl-parallel${PYTHONPATH:+:$PYTHONPATH}"
export PROJECT_ROOT="/home3/swlc12/msc-groundwater-gwl-parallel"
export DATA_ROOT="/home3/swlc12/msc-groundwater-gwl-data"
export RESULTS_ROOT="/home3/swlc12/msc-groundwater-gwl-results"

# let BLAS/PyTorch use all 4 cores
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4


cd /home3/swlc12/msc-groundwater-gwl-parallel

# Just run the driver. No ray start/stop, no worker.
python src/optimisation/main_training_optimiser.py