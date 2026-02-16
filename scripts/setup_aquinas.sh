#!/bin/bash
# Setup vernal on aquinas server
# Run from project root: bash scripts/setup_aquinas.sh
set -e

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

echo ">>> Setting up vernal in $ROOT"
echo ""

# 1. Python environment (venv)
if [ ! -d .venv ]; then
    echo ">>> Creating virtualenv"
    python3 -m venv .venv
fi
source .venv/bin/activate

# 2. Upgrade pip
pip install --upgrade pip

# 3. Install dependencies (CPU PyTorch by default)
# For GPU: after setup, run: pip install torch --index-url https://download.pytorch.org/whl/cu118
echo ">>> Installing dependencies"
pip install -r requirements.txt
pip install 'numpy<2'  # rnaglib/forgi need numpy 1.x (downgrade after deps)

# 3b. Fix DGL graphbolt for PyTorch 2.3.x (symlink 2.2.1 -> 2.3.1)
GB_DIR=$(find .venv -path '*dgl/graphbolt' -type d 2>/dev/null | head -1)
if [ -n "$GB_DIR" ] && [ -f "$GB_DIR/libgraphbolt_pytorch_2.2.1.so" ] && [ ! -f "$GB_DIR/libgraphbolt_pytorch_2.3.1.so" ]; then
    ln -sf libgraphbolt_pytorch_2.2.1.so "$GB_DIR/libgraphbolt_pytorch_2.3.1.so"
    echo ">>> Fixed DGL graphbolt symlink"
fi

# 4. Create data dirs
mkdir -p data/graphs data/annotated data/pdb results/mggs results/trained_models

# 5. Verify
echo ""
echo ">>> Verifying installation"
python -c "
import torch, dgl, networkx, rnaglib
print('torch:', torch.__version__)
print('dgl:', dgl.__version__)
print('CUDA available:', torch.cuda.is_available())
print('rnaglib: OK')
"

echo ""
echo ">>> Setup complete. Activate with: source .venv/bin/activate"
echo ">>> Run full pipeline: bash run_full_pipeline.sh"
echo ">>> Or step by step: python prepare_data/main.py -n rnaglib_full --source rnaglib"
