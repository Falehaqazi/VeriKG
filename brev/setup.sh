#!/usr/bin/env bash
# Setup script for the VeriKG Brev Launchable.
# Installs VeriKG, downloads the prebuilt graph and starts the app on port 8501.
# Set BUILD_FROM_SOURCE=1 to pull Pharos and rebuild everything instead (about 1.5 h).
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
if python3 -c "import torch" 2>/dev/null; then
  python3 -m pip install torch_geometric
else
  python3 -m pip install -r requirements-gnn.txt || echo "PyTorch not installed: GNN steps disabled, app still works"
fi

if [ "${BUILD_FROM_SOURCE:-0}" = "1" ]; then
  python3 -m kgc pull
  python3 -m kgc all
  python3 -m kgc enrich
  python3 -m kgc prune
  python3 -m kgc split-scaffold
  python3 -m kgc benchmark
  python3 -m kgc gnn
  python3 -m kgc gnn --split scaffold --configs pharos,pharos_fp,verikg
  python3 -m kgc predict
else
  python3 -m kgc fetch-data
fi

nohup python3 -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0 \
  --server.headless true > verikg_app.log 2>&1 &
echo "VeriKG app starting on port 8501 (log: verikg_app.log)"
