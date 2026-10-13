#!/usr/bin/env bash
# Phase 4-5 only (SAC deployment, report, animation), reusing the Phase 2-3 results.
# Extra arguments are passed to every script, e.g.
#   bash scripts/server/run_deployment.sh --config configs/default.yaml configs/paper_snr.yaml
set -euo pipefail
cd "$(dirname "$0")/../.."

# shellcheck disable=SC1091
source "${CONDA_DIR:-$HOME/miniforge3}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-uavsec}"
export CUDA_VISIBLE_DEVICES="${GPU:-2}"
python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 'CUDA is not available - check torch/driver')"

python scripts/exp_deployment.py "$@"
python scripts/study_deployment_sensitivity.py "$@" --set opt.restarts=4 opt.iters=150 \
    --vary system.path_loss_log=log10,ln system.distance_model=3d,horizontal system.array_axis=x,y
python scripts/make_report.py "$@"
python scripts/make_animation.py "$@"
