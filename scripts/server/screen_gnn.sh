#!/usr/bin/env bash
# Screen one GNN variant on one GPU (server side): train 3 seeds with study_gnn_optimum.py
# (K = 8 vs OPT, Fig. 7 optimum), then evaluate each checkpoint on the Fig. 9a/9b sweeps.
#
#   GPU=2 bash scripts/server/screen_gnn.sh outputs/studies_r14/s1 gnn.aggregate_self=true
#   GPU=3 bash scripts/server/screen_gnn.sh outputs/studies_r14/base          # no overrides = R13
set -euo pipefail
cd "$(dirname "$0")/../.."

OUT="$1"; shift
# shellcheck disable=SC1091
source "${CONDA_DIR:-$HOME/miniforge3}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-uavsec}"
python -c "import uavsec" 2>/dev/null || pip install -q -e . --no-deps
export CUDA_VISIBLE_DEVICES="${GPU:-2}"

CFG=(--config configs/default.yaml configs/calibrated.yaml)
python scripts/study_gnn_optimum.py "${CFG[@]}" --seeds ${SEEDS:-0 1 2} --set output_dir="$OUT" "$@"
RUN="$(ls -d "$OUT"/gnn_optimum_* | sort | tail -n 1)"
for ckpt in "$RUN"/ckpt/gnn_seed*.pt; do
    seed="$(basename "$ckpt" .pt)"
    python scripts/exp_beamforming.py "${CFG[@]}" --only users power --no-mlp --gnn "$ckpt" \
        --set output_dir="$OUT/eval_$seed" "$@"
done
echo "done: $OUT"
