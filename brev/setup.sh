#!/usr/bin/env bash
# Setup script for the VeriKG Brev Launchable.
# Installs VeriKG, downloads the prebuilt graph and starts the app on port 8501.
# Set BUILD_FROM_SOURCE=1 to pull Pharos and rebuild everything instead (about 1.5 h).
set -euo pipefail
cd "$(dirname "$0")/.."

# Pick an interpreter that can actually install packages. Brev builds a uv
# virtualenv at ~/.venv and only activates it for interactive shells, so a
# startup script sees the bare system python3 - which on Ubuntu 24 ships no
# pip and refuses installs anyway (PEP 668). Prefer the venv, fall back to
# whatever python3 is, and make sure pip exists before using it.
if [ -x "$HOME/.venv/bin/python" ]; then
  PY="$HOME/.venv/bin/python"
else
  PY="$(command -v python3)"
fi
export PIP_BREAK_SYSTEM_PACKAGES=1
if ! "$PY" -m pip --version >/dev/null 2>&1; then
  "$PY" -m ensurepip --upgrade >/dev/null 2>&1 \
    || { command -v uv >/dev/null 2>&1 && uv pip install --python "$PY" pip; } \
    || { echo "no pip for $PY"; exit 1; }
fi
echo "VeriKG installing with $PY ($("$PY" --version 2>&1))"

"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r requirements.txt
if "$PY" -c "import torch" 2>/dev/null; then
  "$PY" -m pip install torch_geometric
else
  "$PY" -m pip install -r requirements-gnn.txt || echo "PyTorch not installed: GNN steps disabled, app still works"
fi

if [ "${BUILD_FROM_SOURCE:-0}" = "1" ]; then
  "$PY" -m kgc pull
  "$PY" -m kgc all
  "$PY" -m kgc enrich
  "$PY" -m kgc prune
  "$PY" -m kgc split-scaffold
  "$PY" -m kgc benchmark
  "$PY" -m kgc gnn
  "$PY" -m kgc gnn --split scaffold --configs pharos,pharos_fp,verikg
  "$PY" -m kgc predict
else
  "$PY" -m kgc fetch-data
fi

nohup "$PY" -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0 \
  --server.headless true > verikg_app.log 2>&1 &
echo "VeriKG app starting on port 8501 (log: verikg_app.log)"
