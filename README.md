# uav-secure

Reproduction of **"Deep Graph Reinforcement Learning for UAV-Enabled Multi-User Secure Communications"**
(X. Tang et al., *IEEE Transactions on Mobile Computing*, vol. 24, no. 9, 2025), implemented from scratch in PyTorch.

The paper solves the joint problem (7)-(9) in two layers:

| Layer | What it decides | Method | Code |
|---|---|---|---|
| Inner | Security beamforming {w_k} at a fixed UAV position | GNN, unsupervised loss (20) | [`models/gnn.py`](src/uavsec/models/gnn.py), [`training/beamformer_trainer.py`](src/uavsec/training/beamformer_trainer.py) |
| Outer | UAV horizontal position q_U | SAC, reward = GNN sum secrecy rate | [`envs/uav_env.py`](src/uavsec/envs/uav_env.py), [`rl/sac_trainer.py`](src/uavsec/rl/sac_trainer.py) |

## Layout

```
configs/default.yaml      Table I of the paper + clearly marked assumptions (single source of truth)
configs/quick.yaml        smoke-test overrides (minutes on a CPU)
src/uavsec/
  channel.py              geometry, Rician channels, path loss (Sec. III / VI)
  secrecy.py              rates and secrecy rate, eqs. (4)-(6), (19)
  models/                 GNN (eqs. 12-19) and MLP baseline
  training/               beamformer training loop (eq. 20)
  baselines/              OPT (multi-start projected gradient), MRT/ZF, deployment heuristics, grid search
  envs/, rl/              MDP (eqs. 26-28) and SAC (eqs. 29-36, Alg. 1)
  evaluation/             beamforming benchmark harness
  plotting.py             one function per paper figure
  report.py, animation.py report generator, trajectory animation
scripts/                  experiment entry points (one per phase)
scripts/server/           environment setup, code sync, run-all
tests/                    pytest suite (rates vs numpy reference, permutation equivariance, env, heuristics)
```

## Setup

Local (CPU is enough for tests and `--quick` runs):

```bash
pip install -e ".[dev]"
python -m pytest
```

GPU server (GTX 1080 Ti, CUDA driver >= 525): `bash scripts/server/setup_env.sh` installs Miniforge and a
`uavsec` env with `torch==2.5.1+cu121` (the last builds that still include Pascal/sm_61 kernels are cu118-cu126).

## Running

Every experiment writes to `outputs/runs/<experiment>_<timestamp>/` with `config.yaml`, `log.txt`,
CSV tables, checkpoints, figures and `metrics.json` (with git commit, seed and wall time).
Each command accepts `--config`, `--set key=value ...` and `--quick`.

| Phase | Command | Reproduces |
|---|---|---|
| 2 | `python scripts/exp_convergence.py` | Fig. 4 (GNN vs MLP), Fig. 5 (transfer 8 -> 12 users) |
| 3 | `python scripts/exp_beamforming.py` | Fig. 9a-c (K / power / noise sweeps), Fig. 10 (CDF), Fig. 11 (run time) |
| 4 | `python scripts/exp_deployment.py` | Fig. 6 (SAC convergence), Fig. 7 (trajectory vs optimum), Fig. 8 (heuristics) |
| 5 | `python scripts/make_report.py` | `outputs/report/report.md` + all figures (PNG/PDF) + `summary.json` |
| 5 | `python scripts/make_animation.py` | `animation/uav_trajectory.gif/.mp4` of the learned deployment |

Phases 3 and 4 automatically use the GNN checkpoint of the latest Phase 2 run (or pass `--gnn path`).

Main results (calibrated channel): add `--config configs/default.yaml configs/calibrated.yaml` to every command;
they are written to `outputs/runs_calibrated/` and the report to `outputs/runs_calibrated/report/report.md`.

On the server:

```bash
scripts/server/sync.sh push                       # from the local machine
ssh ricardo@140.123.104.113
cd uav-secure && tmux new -d -s uavsec 'GPU=2 bash scripts/server/run_all.sh 2>&1 | tee outputs/run_all.log'
scripts/server/sync.sh pull                       # fetch outputs back
```

## Documentation

- [docs/MODEL_HISTORY.md](docs/MODEL_HISTORY.md): architecture, timeline of every change with reason and evidence, result snapshots, open issues.
- [docs/PARAMETERS.md](docs/PARAMETERS.md) / [docs/parameters.json](docs/parameters.json): current parameters, each marked as paper value, assumption or calibrated choice, with the alternatives already tried.
- Regenerate after changing configs: `python scripts/export_docs.py` (history is kept in `docs/history.json`).

## Deviations from the paper

- **OPT baseline.** The optimisation scheme of [45] is not described in the paper. It is replaced by per-instance
  projected-gradient ascent on the sum secrecy rate with multiple restarts (MRT, ZF, random).
- **Unreported hyper-parameters** (GNN width, steps per episode, SAC action scale, ...) are set in
  `configs/default.yaml` and marked `# assumption`.
- **Eq. (19)** is applied with sqrt(P^max), so that the power constraint holds with equality.
- **Channel interpretation (main results use [`configs/calibrated.yaml`](configs/calibrated.yaml)).**
  Every Table I value is kept. The paper writes the path loss as "30 + 22 log(d)" without saying which
  logarithm or which distance. With log10 and the 3-D distance (as printed), the rate map over the area is
  almost flat: every heuristic reaches over 90 % of the optimum, while the paper's Fig. 8 shows 39-61 %, and
  the optimum is ill-defined (about 50 m from the paper's). Reading log as the natural logarithm, with the
  horizontal UAV-node distance, reproduces the paper's deployment results: the GNN-based optimum lies 2.8 m
  from the paper's Fig. 7 optimum (all three training seeds are within 8.2 m). The evidence is in
  `scripts/study_deployment_sensitivity.py` (section 5 of the report). Table I as printed is still run as a
  reference (`outputs/runs/`, `outputs/report/`).
- **Unspecified training details** (all in `configs/default.yaml`, marked `# assumption`): GNN hidden width 256
  with 50 fresh mini-batches per epoch (width 128 under-fits and its rate map drifts, see
  `scripts/study_gnn_optimum.py`), gradient-norm clipping, log-compressed channel part of the SAC state,
  SAC reward scaled by 0.1 for training (figures report the unscaled R_sec), and the best evaluated SAC
  policy is kept.
- **Fig. 7 optimum** is found by exhaustive search with the GNN as beamformer, because problem (23)-(25)
  fixes W = GNN output (eq. (24)). The OPT-based optimum is reported alongside for reference.
