#!/usr/bin/env bash
# One-time environment setup on the GPU server (Ubuntu, no root needed).
# Installs Miniforge into ~/miniforge3 and creates the `uavsec` conda env.
#
# torch 2.5.1+cu121 is pinned on purpose: it still ships sm_61 kernels for the
# GTX 1080 Ti (Pascal) and runs on driver 535 (CUDA 12.2).
set -euo pipefail

CONDA_DIR="${CONDA_DIR:-$HOME/miniforge3}"
ENV_NAME="${ENV_NAME:-uavsec}"
PY_VERSION="3.11"

if [ ! -x "$CONDA_DIR/bin/conda" ]; then
    echo "[setup] installing Miniforge into $CONDA_DIR"
    installer="$(mktemp -d)/miniforge.sh"
    curl -fsSL -o "$installer" \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    bash "$installer" -b -p "$CONDA_DIR"
    rm -f "$installer"
fi

# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"

if ! conda env list | grep -qE "^$ENV_NAME\s"; then
    echo "[setup] creating env $ENV_NAME (python $PY_VERSION)"
    conda create -y -n "$ENV_NAME" "python=$PY_VERSION"
fi
conda activate "$ENV_NAME"

pip install --upgrade pip
# torch is pinned in the same command so no other package can upgrade it to a
# build that needs a newer driver.
pip install "torch==2.5.1" "stable-baselines3>=2.5" "gymnasium>=1.0,<1.3" \
    numpy scipy pandas matplotlib pyyaml tqdm pytest imageio imageio-ffmpeg tensorboard \
    --extra-index-url https://download.pytorch.org/whl/cu121

python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("devices:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    x = torch.randn(1024, 1024, device="cuda")
    print("matmul ok:", float((x @ x).sum()) == float((x @ x).sum()))
EOF
echo "[setup] done. Activate with: source $CONDA_DIR/etc/profile.d/conda.sh && conda activate $ENV_NAME"
