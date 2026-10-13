#!/usr/bin/env bash
# Run every phase in order on one GPU (server side). Extra arguments are passed
# to each experiment, e.g. `--quick` or `--set system.noise_power_w=1.2e-10`.
#
#   GPU=2 bash scripts/server/run_all.sh                 # full reproduction
#   tmux new -d -s uavsec 'bash scripts/server/run_all.sh 2>&1 | tee outputs/run_all.log'
set -euo pipefail
cd "$(dirname "$0")/../.."

# shellcheck disable=SC1091
source "${CONDA_DIR:-$HOME/miniforge3}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-uavsec}"
# Install only when missing: concurrent runs must not reinstall the package at the same time.
python -c "import uavsec" 2>/dev/null || pip install -q -e . --no-deps
export CUDA_VISIBLE_DEVICES="${GPU:-2}"
python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 'CUDA is not available - check torch/driver')"

python -m pytest -q
python scripts/exp_convergence.py "$@"     # Phase 2: Fig. 4-5
python scripts/exp_beamforming.py "$@"     # Phase 3: Fig. 9-11
python scripts/exp_deployment.py "$@"      # Phase 4: Fig. 6-8
python scripts/make_report.py "$@"         # Phase 5: report + figures
python scripts/make_animation.py "$@"      # Phase 5: trajectory animation
