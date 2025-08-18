#!/bin/bash

# Instructing SLURM to locate and assign X number of nodes with Y number of cores in each node.
# X,Y are integers. Refer to table for various combinations. X will almost always be 1.
#SBATCH -N 1
#SBATCH -c 4

# Governs the run time limit and resource limit for the job. Pick values from the partition and QOS tables.
#SBATCH --gres=gpu:turing:1
#SBATCH -p tpg-gpu-small
#SBATCH --qos=short
#SBATCH -t 02-00:00:00
#SBATCH --mem=20G

#SBATCH --job-name=GWL_pipeline_run
#SBATCH -o /home3/swlc12/msc-groundwater-gwl-parallel/slurm/logs/slurm-%j.out

#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=swlc12@durham.ac.uk

# Faily immediately with errors, undefined vars or if any command fails (not just last)
set -euo pipefail

# Source the bash profile (required to use the module command, ignore warning blocks here)
set +u
source /etc/profile
set -u

# Activate .venv
source /home3/swlc12/msc-groundwater-gwl/.venv_ncc/bin/activate

# Log selected GPU and node before runnign code
echo "Running on $(hostname)"
nvidia-smi || true

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTHONPATH="/home3/swlc12/msc-groundwater-gwl-parallel${PYTHONPATH:+:$PYTHONPATH}"

# Define test and validation stations
TEST="${TEST:-}"
VALS="${VALS:-}"  # colon-separated list, e.g. "coupland:renwick"

# Go to project dir (helps with relative paths/checkpoints)
cd /home3/swlc12/msc-groundwater-gwl-parallel

# Run your program (replace this with your program)
python main_training.py --test "$TEST" --vals "$VALS"