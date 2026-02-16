#!/bin/bash
# Full vernal pipeline: prepare_data -> train_embeddings -> build_motifs
# Uses RNADataset by default for motif building (no -g flag)
set -e
cd "$(dirname "$0")"

NAME="${1:-rnaglib_full}"
MODEL="${2:-rnaglib_full_model}"
MGG="${3:-rnaglib_full_mgg}"

echo ">>> Full pipeline: name=$NAME model=$MODEL mgg=$MGG"
echo ""

# 1. Prepare data (remove existing to avoid "name already taken")
echo ">>> Step 1: prepare_data"
rm -rf "data/graphs/$NAME" "data/annotated/$NAME"
python prepare_data/main.py -n "$NAME" --source rnaglib
echo ""

# 2. Train embeddings
echo ">>> Step 2: train_embeddings"
python train_embeddings/main.py train -da "$NAME" -n "$MODEL" -ep 30 -bs 2 -nw 4
echo ""

# 3. Build motifs (RNADataset default, no -g)
echo ">>> Step 3: build_motifs"
mkdir -p results/mggs
python build_motifs/main.py -r "$MODEL" --mgg_name "$MGG" -b -N 200
echo ""

echo ">>> Done. Meta-graph: results/mggs/${MGG}.p"
echo ">>> JSON: results/mggs/${MGG}.json (open visualize_motifs.html to view)"
