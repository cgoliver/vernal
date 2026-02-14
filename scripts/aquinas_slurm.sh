#!/bin/bash
#SBATCH --job-name=vernal
#SBATCH --output=vernal_%j.out
#SBATCH --error=vernal_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Vernal train + build on aquinas (SLURM)
# Prereqs: run setup_aquinas.sh and prepare_data first
# sbatch scripts/aquinas_slurm.sh

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

echo ">>> Vernal train+build on $(hostname)"
echo ">>> GPU: $(nvidia-smi -L 2>/dev/null || echo 'N/A')"
echo ""

# -nw 0 avoids DataLoader multiprocessing pickle errors
python train_embeddings/main.py train -da rnaglib_full -n rnaglib_full_model -ep 30 -bs 4 -nw 0
python build_motifs/main.py -r rnaglib_full_model --mgg_name rnaglib_full_mgg -b -N 200

echo ">>> Done. Check results/mggs/"
