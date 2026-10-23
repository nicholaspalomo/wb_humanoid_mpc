#!/bin/bash
# ==============================================================================
# launch_jupyter.sh — Starts Jupyter for the ACoM SIREN training notebook
# (notebooks/train_acom_siren.ipynb).
#
# Usage:
#   make train-acom-jupyter
#   # or
#   tools/launch_jupyter.sh [PORT]    # PORT defaults to 8888
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SCRIPT_DIR}"

# The Bazel environment, and robotpkg's Pinocchio bindings on PYTHONPATH (setup_env.sh)
if [ -f "${SCRIPT_DIR}/setup_env.sh" ]; then
    source "${SCRIPT_DIR}/setup_env.sh"
fi

# Auto-start VNC server so 3D viewer is immediately accessible
if [ -f "${SCRIPT_DIR}/.devcontainer/start_vnc.sh" ]; then
    "${SCRIPT_DIR}/.devcontainer/start_vnc.sh" 2>/dev/null || true
fi

PORT="${1:-8888}"

echo "=================================================================="
echo "Starting Jupyter for the ACoM SIREN training notebook"
echo "=================================================================="

export PATH="${HOME}/.local/bin:${PATH}"

# Ensure pip & jupyter packages exist in environment
if ! python3 -m jupyter --version &> /dev/null; then
    echo "📦 Setting up Jupyter environment..."
    if ! python3 -m pip --version &> /dev/null; then
        echo "📥 Installing python3-pip..."
        if command -v sudo &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y --no-install-recommends python3-pip python3-venv
        elif [ "$(id -u)" -eq 0 ]; then
            apt-get update && apt-get install -y --no-install-recommends python3-pip python3-venv
        fi
    fi
    echo "📥 Installing JupyterLab, Notebook, IPyWidgets, MuJoCo, and JAX..."
    python3 -m pip install --no-cache-dir --ignore-installed --break-system-packages jupyterlab notebook ipywidgets matplotlib tensorboardX plotly mujoco jax jaxlib optax
    python3 -m ipykernel install --user --name "wb_humanoid_mpc" --display-name "Python 3 (Humanoid MPC)" 2>/dev/null || true
fi

NOTEBOOK_DIR="${SCRIPT_DIR}/notebooks"
mkdir -p "${NOTEBOOK_DIR}"

echo "📁 Notebook directory: ${NOTEBOOK_DIR}"
echo "🌐 Listening on: http://0.0.0.0:${PORT}"
echo "Open 'notebooks/train_acom_siren.ipynb' (ACoM SIREN training) once Jupyter is running."
echo "=================================================================="

# Launch Jupyter Notebook server
if python3 -m jupyter --version &> /dev/null; then
    exec python3 -m jupyter notebook \
        --notebook-dir="${SCRIPT_DIR}" \
        --ip=0.0.0.0 \
        --port="${PORT}" \
        --no-browser \
        --allow-root \
        --ServerApp.token='' \
        --ServerApp.password='' \
        --NotebookApp.token='' \
        --NotebookApp.password=''
elif command -v jupyter &> /dev/null; then
    exec jupyter notebook \
        --notebook-dir="${SCRIPT_DIR}" \
        --ip=0.0.0.0 \
        --port="${PORT}" \
        --no-browser \
        --allow-root \
        --ServerApp.token='' \
        --ServerApp.password='' \
        --NotebookApp.token='' \
        --NotebookApp.password=''
else
    echo "❌ Error: Jupyter is not installed. Please run: pip3 install jupyterlab notebook ipywidgets"
    exit 1
fi
