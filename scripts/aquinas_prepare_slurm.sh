#!/bin/bash
#SBATCH --job-name=vernal_prep
#SBATCH --output=vernal_prep_%j.out
#SBATCH --error=vernal_prep_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Vernal prepare_data on aquinas (SLURM)
# sbatch scripts/aquinas_prepare_slurm.sh

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

echo ">>> Vernal prepare_data on $(hostname)"
python prepare_data/main.py -n rnaglib_full --source rnaglib
echo ">>> Done. Run aquinas_slurm.sh next for train+build."
