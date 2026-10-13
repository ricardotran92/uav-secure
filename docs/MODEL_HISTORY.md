# Model and parameter history

_Reproduction of Tang et al., 'Deep Graph Reinforcement Learning for UAV-Enabled Multi-User Secure Communications' (IEEE TMC 2025)_  
Updated 2026-10-14.

- **Main profile:** configs/default.yaml + configs/calibrated.yaml  ->  outputs/runs_calibrated/
- **Reference profile:** configs/default.yaml (Table I as printed)  ->  outputs/runs/
- Current parameters with provenance markers: [PARAMETERS.md](PARAMETERS.md) (machine-readable: [parameters.json](parameters.json)). Source of this history: [history.json](history.json).

## 1. Architecture

| Block | What it does | Paper |
|---|---|---|
| **Channel (channel.py)** | Rician fading K=10 dB; LoS = half-wavelength ULA steering vector at the UAV; path loss 30 + 22 log(d) dB; channels returned divided by sigma_0 (noise power = 1) for float32 conditioning. | Sec. III, VI |
| **Secrecy rate (secrecy.py)** | R_k = log2(1 + \|h^H w_k\|^2 / (sum_{l!=k} \|h^H w_l\|^2 + 1)); R_sec = (R_k - R_E,k)^+; power normalisation sqrt(P) * W / \|\|W\|\|. | eq. (4)-(6), (19), (21)-(22) |
| **GNN (models/gnn.py)** | K nodes (user + its eavesdropper); D GCN layers; node weights shared; message generation -> max aggregation over the other nodes -> concat -> output MLP; PReLU; embedding e_k updated, channel part kept; final normalisation. | eq. (12)-(19), Fig. 2 |
| **MLP baseline (models/mlp.py)** | Flattened channels of all K pairs -> all K beamformers; input size tied to K (one MLP per K). | 'conventional deep model with pure MLP structure' |
| **Beamformer training (training/beamformer_trainer.py)** | Unsupervised, loss = -mean sum secrecy rate; fresh random layouts, UAV positions and fading every mini-batch; best-validation checkpoint kept. | eq. (20) |
| **OPT baseline (baselines/opt_beamforming.py)** | Per-instance projected-gradient (Adam) ascent on the sum secrecy rate, multi-start (MRT, ZF, random). Stand-in for [45], which the paper does not describe. | substitute for [45] |
| **Deployment MDP (envs/uav_env.py)** | State = UAV position + channel features; action = (dx, dy) in [-1,1]^2 scaled to metres; reward = sum secrecy rate with GNN beamforming at the new position; fresh fading every step. | eq. (26)-(28) |
| **SAC (rl/sac_trainer.py)** | Stable-Baselines3 SAC (twin critics, target networks, auto-tuned temperature) plus periodic deterministic evaluation that keeps the best policy. | eq. (29)-(36), Alg. 1 |
| **Deployment baselines (baselines/deployment.py)** | Area centre, geometric centre, circumcentre (minimum enclosing circle), polygon centroid (convex hull); exhaustive grid search of the expected secrecy rate, scored with GNN beamforming (problem (23)-(25), constraint (24)). | Fig. 7-8 |

## 2. Current state at a glance

**Calibrated / assumed parameters in use:**

- 🔴 `system.path_loss_log` = **`ln`** - the paper writes "log(d)": log10 (standard) | ln
- 🟡 `system.antenna_spacing` = **`0.5`** - assumption: half-wavelength ULA
- 🔴 `system.distance_model` = **`horizontal`** - assumption: path-loss distance, 3d = sqrt(d_h^2 + H^2) | horizontal = d_h
- 🔴 `system.min_distance_m` = **`20`** - assumption: path-loss distance is clamped below at this value
- 🔴 `system.array_axis` = **`x`** - assumption: ULA orientation in the horizontal plane (x | y)
- 🟡 `gnn.hidden_dim` = **`512`** - assumption (paper gives no width; 128/256 under-fit, 1024 collapses at lr 0.005)
- 🟡 `gnn.batches_per_epoch` = **`50`** - assumption: fresh mini-batches per epoch
- 🟡 `gnn.val_samples` = **`2048`** - assumption: fixed validation set
- 🟡 `gnn.grad_clip` = **`1`** - assumption: gradient-norm clipping (training diverged without it)
- 🟡 `gnn.lr_schedule` = **`cosine`** - assumption: starts at the Table I lr 0.005, decays to 1% (paper gives no schedule)
- 🟡 `gnn.input_transform` = **`linear`** - assumption: channel input scaling, linear | log (log-compressed magnitude, phase kept); also used by the MLP
- 🟡 `gnn.uav_near_user_frac` = **`0`** - assumption: share of training samples with the UAV around a random user (rest uniform)
- 🟡 `gnn.uav_near_user_std_m` = **`30`** - assumption: spread (m) of those UAV positions around the user
- 🟡 `mlp.hidden_dims` = **`[512, 512, 512]`** - assumption
- 🟡 `mlp.lr` = **`0.001`** - assumption (0.005 diverges for the wider MLP)
- 🟡 `sac.episode_length` = **`175`** - assumption: Fig. 6 reward / secrecy-rate ratio suggests ~170-180 steps
- 🟡 `sac.tau` = **`0.005`** - assumption: SB3 default soft-update rate
- 🟡 `sac.learning_starts` = **`1000`** - assumption
- 🟡 `sac.net_arch` = **`[256, 256]`** - assumption
- 🟡 `sac.max_step_m` = **`2.5`** - assumption: action [-1,1]^2 scaled to metres (smoother trajectory than 5 m)
- 🟡 `sac.obs_transform` = **`log`** - assumption: channel part of the state as sign(x) log(1 + |x|) (linear | log)
- 🟡 `sac.reward_scale` = **`0.1`** - assumption: SAC trains on 0.1 x R_sec; logs and figures use the unscaled R_sec
- 🟡 `sac.eval_every_episodes` = **`10`** - assumption: deterministic evaluation interval; the best policy is kept
- 🟡 `sac.gradient_steps` = **`2`** - assumption: SAC gradient updates per environment step
- 🟡 `sac.eval_episodes` = **`3`** - assumption: deterministic episodes averaged per evaluation

**Main-profile results:**

- GNN 10.25 +- 0.10 vs MLP 7.79 +- 0.11 bps/Hz (validation).
- K=8, 1 W: GNN 12.14 / OPT 16.75 / MLP 8.93 bps/Hz (GNN = 72% of OPT).
- Fig. 7 optimum (58, 142) vs paper (56, 140): **2.8 m**; SAC ends 20.0 m from the optimum.
- Graph-DRL reaches 92.8% of the optimum, +17% over the best heuristic.

## 3. Timeline of changes

| ID | Area | Change | Status |
|---|---|---|---|
| C01 | infra | Project scaffold, typed YAML configuration, run directories | ✅ active |
| C02 | gnn | Initial model choices where the paper is silent | ✅ active |
| C03 | gnn | PReLU on the last axis | ✅ active |
| C04 | infra | Server environment: torch pinned, CUDA check | ✅ active |
| C05 | gnn | Gradient-norm clipping (1.0) for beamformer training | ✅ active |
| C06 | system_model | Hypothesis A: noise power 30 dB higher (configs/paper_snr.yaml) | ⏭️ superseded |
| C07 | system_model | Fixed SAC layout digitised from Fig. 7 | ✅ active |
| C08 | system_model | Channel interpretation: natural log + horizontal distance (configs/calibrated.yaml) | ✅ active |
| C09 | evaluation | Exhaustive search for the Fig. 7 optimum uses the GNN | ✅ active |
| C10 | gnn | GNN capacity: hidden 128 -> 256, batches per epoch 20 -> 50 | ⏭️ superseded |
| C11 | gnn | Log-compressed channel magnitude as GNN input | ↩️ reverted |
| C12 | sac | SAC: log-compressed state, reward scale 0.1, keep the best evaluated policy | ✅ active |
| C13 | infra | Smaller fixes | ✅ active |
| C14 | gnn | Diagnosis: the GNN-based optimum jumped onto spikes right above single users | ✅ active |
| C15 | system_model | Minimum horizontal path-loss distance of 20 m (calibrated profile) | ✅ active |
| C16 | gnn | GNN: hidden 512 and cosine learning-rate schedule starting at 0.005 | ✅ active |
| C17 | sac | SAC: 2 gradient steps per step, 2.5 m steps, 175 steps per episode, 3-episode evaluation | ✅ active |
| C18 | gnn | Training UAV positions sampled near users (50%) | ✖ rejected |
| C19 | infra | Memory-efficient max aggregation | ✅ active |

### C01 - Project scaffold, typed YAML configuration, run directories  ·  ✅ active

- **Before:** Empty repository.
- **After:** Package src/uavsec, configs/default.yaml as the single source of truth (Table I + clearly marked assumptions), every run writes config.yaml, log.txt, metrics.json with git hash, seed and wall time.
- **Why:** Traceability: every number in the report must come from a recorded run.
- **Evidence:** 19 unit tests (rates vs numpy reference, permutation equivariance, power constraint, env, heuristics).
- **Commits:** `ff14c54`, `15d539b`

### C02 - Initial model choices where the paper is silent  ·  ✅ active

- **Before:** -
- **After:** GNN hidden width 128, weights shared across nodes, max aggregation, PReLU; MLP 3x512; OPT 16 restarts x 500 iterations; SAC episode length 100, max step 5 m, networks 2x256; area-centre start; eavesdropper 20 m from its user at a random bearing.
- **Why:** Paper gives no widths, episode length, step size or OPT details.
- **Evidence:** Local CPU sanity check: GNN 12.6 vs MLP 6.4 bps/Hz at a 30 dB lower SNR (30 epochs).
- **Commits:** `ff14c54`

### C03 - PReLU on the last axis  ·  ✅ active

- **Before:** torch nn.PReLU(num_parameters) on tensors shaped (B, K, H) -> shape mismatch error.
- **After:** FeaturePReLU: one learnable slope per feature on the last axis.
- **Why:** Bug found by the first tests.
- **Evidence:** Tests passed 15/15 afterwards.
- **Commits:** `ff14c54`

### C04 - Server environment: torch pinned, CUDA check  ·  ✅ active

- **Before:** Upgrading stable-baselines3 silently replaced torch 2.5.1+cu121 by torch 2.14 (CUDA 13); driver 535 (CUDA 12.2) cannot run it, so training fell back to CPU.
- **After:** torch==2.5.1 pinned in the same pip command; run_all.sh aborts if CUDA is not available; concurrent runs no longer reinstall the package.
- **Why:** GPU 2 showed 0% utilisation while the process used 2700% CPU.
- **Evidence:** After the fix GPU 2 at 97%.
- **Commits:** `c39f9bf`, `af2b07a`

### C05 - Gradient-norm clipping (1.0) for beamformer training  ·  ✅ active

- **Before:** No clipping; Adam lr 0.005 (paper's value).
- **After:** gnn.grad_clip = 1.0.
- **Why:** GNN reached 17.3 bps/Hz and then collapsed to about 5 near epoch 125.
- **Evidence:** Clipped runs stay stable for all 300 epochs (about 18 bps/Hz at 128 width).
- **Commits:** `13b48ec`

### C06 - Hypothesis A: noise power 30 dB higher (configs/paper_snr.yaml)  ·  ⏭️ superseded

- **Before:** Table I as printed gives a per-antenna SNR of about 55 dB; zero-forcing alone reaches 86-90 bps/Hz at K=8, far above the paper's OPT curve (about 15-18).
- **After:** Re-run everything with noise 1.2e-10 W (a W/mW slip).
- **Why:** Paper's Fig. 9b looks consistent with an SNR about 30 dB lower.
- **Evidence:** At 1 W, K=8: GNN 20.8 / OPT 35.1 / MLP 12.4 (GNN = 59% of OPT). But the deployment optimum was still 24 m off and the rate map stayed flat, so this hypothesis does not explain Fig. 7/8. Dropped as the main profile after the user pointed out that 30+22log(d) is the official formula.
- **Impact:** Kept only as an extra reference (configs/paper_snr.yaml).
- **Commits:** `5a15530`

### C07 - Fixed SAC layout digitised from Fig. 7  ·  ✅ active

- **Before:** Random user/eavesdropper layout (seed 2025).
- **After:** configs/layouts/paper_fig7.yaml: 8 users + 8 eavesdroppers read off the figure (+-1 m), including the user hidden behind the legend at (137, 143); UAV starts at (100, 100). Paper reference optimum (56, 140) and Fig. 8 bar heights stored in the same file for comparison only.
- **Why:** Needed to compare the learned deployment with the paper's.
- **Evidence:** User approved the layout figure (outputs/layout/layout_check.png). Pairing of eavesdroppers to users is an inference (paper does not say). A test of both pairings of the U5/U6 cluster could not tell them apart, so the one closest to 'about 20 m' was kept.
- **Commits:** `5ff8ff3`

### C08 - Channel interpretation: natural log + horizontal distance (configs/calibrated.yaml)  ·  ✅ active

- **Before:** log10 and 3-D distance in the path loss (the 'standard' reading). Deployment optimum 45-62 m from the paper's (56, 140); heuristics reach over 90% of the optimum (paper Fig. 8: 39-61%); rate map almost flat.
- **After:** system.path_loss_log = ln, system.distance_model = horizontal, system.array_axis = x. All Table I numbers unchanged.
- **Why:** The paper writes 'log(d)' without base and does not say which distance. A sensitivity study over 8 combinations (log base x distance x ULA axis, plus noise levels) found the combination that reproduces three independent paper results at once: Fig. 7 optimum location, Fig. 8 heuristic/optimum profile, secrecy-rate level.
- **Evidence:** OPT beamformer, optimum distance to paper: ln+3D+x 10 m, ln+horizontal+x 17 m, ln+horizontal+y 25 m, log10 variants 44-62 m. Fig. 8 profile error: ln+horizontal+x 0.15 (best), log10 variants 0.42-0.48. Rate level 30 bps/Hz vs paper 23. Chosen by the user after reviewing the study (scripts/study_deployment_sensitivity.py, report section 5).
- **Impact:** Main profile. Table I as printed is kept as a reference run.
- **Commits:** `152b4cc`, `3493cc3`, `7f5a48d`, `5daf093`

### C09 - Exhaustive search for the Fig. 7 optimum uses the GNN  ·  ✅ active

- **Before:** First changed to the OPT beamformer (joint optimum of (7)-(9)).
- **After:** Default eval.search_beamformer = gnn (problem (23)-(25) fixes W = GNN output, eq. (24)); the OPT-based optimum is always computed and reported next to it.
- **Why:** User challenged whether the code still follows the paper; constraint (24) requires GNN beamforming inside the deployment problem.
- **Evidence:** Consequence: the optimum now depends on GNN quality, which led to C10.
- **Commits:** `a2f3ea2`, `10d5143`, `8e04a16`

### C10 - GNN capacity: hidden 128 -> 256, batches per epoch 20 -> 50  ·  ⏭️ superseded

- **Before:** hidden 128, 20 mini-batches per epoch: GNN reaches 71% of OPT on the calibrated model; its rate map put the optimum 45 m from the paper's.
- **After:** gnn.hidden_dim = 256, gnn.batches_per_epoch = 50 (paper gives neither).
- **Why:** Better GNN -> rate map closer to the true one. Architecture, loss, lr 0.005, 300 epochs, batch 512 unchanged.
- **Evidence:** 128/20: 71% of OPT, optimum 45 m off. 256/20: 74%, 8 m. 128/50: 75%, 9 m. 256/50: 76%, 0 m. Three independent seeds at 256/50 (script study_gnn_optimum.py): 2.8 m, 2.8 m, 80 m (the third is a weaker GNN and is never selected because the deployed GNN is the best-validation seed).
- **Impact:** Training time 3 min -> 11 min per seed on GPU. Superseded by C16 (hidden 512 + cosine).
- **Commits:** `aa6fd59`

### C11 - Log-compressed channel magnitude as GNN input  ·  ↩️ reverted

- **Before:** Linear scaling sqrt(P/SNR_ref) * h.
- **After:** Tried log(1+|h|); reverted.
- **Why:** Channel gain spans decades under the horizontal-distance path loss (feature 99th percentile about 67, median 1.8).
- **Evidence:** No gain: linear GNN 8.93 / MLP 6.78 vs log GNN 8.90 / MLP 6.63 bps/Hz (100 epochs). Re-tested on 2026-10-14 in the dense user cluster (G8: 512 + log -> 5.6 val, broken; G9: 256 + log -> 73% of OPT, same as linear). Not adopted.
- **Commits:** `571480d`, `78e1a0b`

### C12 - SAC: log-compressed state, reward scale 0.1, keep the best evaluated policy  ·  ✅ active

- **Before:** Linear state, raw reward (about 14-28), last policy kept. SAC reached about 18 bps/Hz in the first 100 episodes and then degraded; final point (121, 118) was 63% of the optimum and 20% below the best heuristic.
- **After:** sac.obs_transform = log, sac.reward_scale = 0.1 (training only; logs and figures use the unscaled R_sec), sac.eval_every_episodes = 10 with the best deterministic policy restored at the end. Extended by C17.
- **Why:** Large channel features and reward scale hurt the critics; SAC's last iterate is noisy.
- **Evidence:** Calibrated: final point (73.6, 154.5), 92.8% of the optimum, +17% over the best heuristic. Table I: 93.1%, +4.8%.
- **Commits:** `99d03f5`

### C13 - Smaller fixes  ·  ✅ active

- **Before:** latest_run('deployment') also matched 'deployment_sensitivity_*'; '--set' given twice kept only the last list; report crashed when metrics lacked a field.
- **After:** Exact timestamp match; --set accumulates; report tolerates older metrics.
- **Why:** Found while running the pipelines.
- **Evidence:** Smoke tests and full runs.
- **Commits:** `61e2ef7`, `a2f3ea2`

### C14 - Diagnosis: the GNN-based optimum jumped onto spikes right above single users  ·  ✅ active

- **Before:** Better GNNs (cosine LR, more width or data) reached 77-80% of OPT but put the Fig. 7 optimum right above user U2 (136, 144) or U1 (88, 158), 37-80 m from the paper's.
- **After:** Both candidates measured with 2048 fading draws: under OPT the paper's cluster (29.3 bps/Hz) beats the point above U2 (25.5); the GNNs reached 101-106% of OPT above U2 but only 70-77% in the cluster.
- **Why:** Find out whether the paper's optimum or the spike is the true optimum.
- **Evidence:** Root cause: with the horizontal-distance path loss clamped at 1 m, a UAV right above a user gives that user about 99 dB SNR (impossible at 100 m altitude). The paper's optimum is 21 m from the nearest user, so its model has no such spikes.

### C15 - Minimum horizontal path-loss distance of 20 m (calibrated profile)  ·  ✅ active

- **Before:** Hard-coded clamp at 1 m.
- **After:** system.min_distance_m (default 1.0); configs/calibrated.yaml sets 20 m.
- **Why:** Do not extrapolate the empirical 30 + 22 ln(d) model below 20 m horizontal distance (the paper's user-eavesdropper scale); this removes the unphysical spikes.
- **Evidence:** OPT, 2 m grid: 1 m -> optimum 18 m from paper, profile error 0.15; 10 m -> same; 20 m -> 16 m, paper point 96% of max, profile 0.16, rate 29.9; 50 m -> 33 m; 100 m -> 37 m. GNN (256, cosine) at 20 m: 3 seeds give 81-82% of OPT and optima 7.2 / 8.2 / 10.8 m from the paper's.

### C16 - GNN: hidden 512 and cosine learning-rate schedule starting at 0.005  ·  ✅ active

- **Before:** hidden 256, constant lr 0.005 (about 72% of OPT).
- **After:** gnn.hidden_dim = 512, gnn.lr_schedule = cosine (starts at the Table I value 0.005, decays to 1%). Still 5 GCN layers, batch 512, 300 episodes.
- **Why:** Constant lr plateaued at about 72% of OPT whatever the width or data; the paper gives no schedule. The user approved the schedule on 2026-10-14.
- **Evidence:** Constant lr: 256 -> 72%, 512 -> similar (G6), 256 + near-user sampling -> 71% (G10). Cosine: 256 -> 77-82%, 512 -> 80-84%. Hidden 1024 collapsed at lr 0.005 (all secrecy rates clamped to 0, zero gradient). Final candidate 512 + cosine + 20 m: 84% of OPT, optimum exactly at (56, 140).

### C17 - SAC: 2 gradient steps per step, 2.5 m steps, 175 steps per episode, 3-episode evaluation  ·  ✅ active

- **Before:** 1 gradient step, 5 m steps, 100 steps per episode, 1 evaluation episode: SAC stopped 20 m from the optimum and jittered about 5 m per step.
- **After:** sac.gradient_steps = 2, sac.max_step_m = 2.5, sac.episode_length = 175, sac.eval_episodes = 3. Table I SAC values (lr, 500 episodes, mini-batch 64, buffer 1e6, discount 0.99) unchanged.
- **Why:** More updates per sample, smaller steps for a smoother path, horizon long enough to reach the optimum. Fig. 6 of the paper (episode reward about 4.4e3 at about 25 bps/Hz) implies about 170-180 steps per episode.
- **Evidence:** With the previous GNN: S1 (2 grad steps) 10.5 m from paper; S3 (+3-episode eval) 5.6 m; S2 (T=200) 13.4 m; S4 (2.5 m, T=200, 1 grad step) 56.9 m; S5 (2.5 m, T=100) 57.7 m (cannot reach); S6 (2 grad steps, 2.5 m, T=175) 3.9 m with 1.0 m jitter.

### C18 - Training UAV positions sampled near users (50%)  ·  ✖ rejected

- **Before:** UAV uniform in the area during GNN training.
- **After:** Option gnn.uav_near_user_frac kept at 0.0.
- **Why:** Hypothesis: the GNN is weak where SAC places the UAV.
- **Evidence:** G10 (constant lr) 71% of OPT; G12/G13 (cosine) 77-79% with optima on the spike above U1 (37 m). No gain over uniform sampling.

### C19 - Memory-efficient max aggregation  ·  ✅ active

- **Before:** The max over the other nodes built a B x K x K x H tensor (2 GB for 32k samples): out of memory on the shared GPU.
- **After:** Top-2 trick, O(K) memory, bit-identical results (checked for K = 1, 2, 5, 12 and ties); evaluation chunks of 8192.
- **Why:** Two SAC runs crashed in the final grid search.
- **Evidence:** Tests pass; outputs identical to the K x K reference.

## 4. Result snapshots

### R1 - Table I as printed, GNN 128/20 (first full run)

```json
{
  "after_changes": [
    "C01",
    "C05"
  ],
  "convergence": {
    "gnn_val": "20.59 +- 2.98",
    "mlp_val": "17.70 +- 0.06",
    "transfer_8_to_12": "25.85 (1-3 epochs to 95%) vs re-training 22.33 (65-82 epochs)"
  },
  "beamforming_K8_1W": {
    "GNN": 25.67,
    "OPT": 90.87,
    "MLP": 18.79,
    "ZF": 89.96
  },
  "deployment": "GNN-surface optimum (120, 122): 66.5 m from the paper's. Random-layout SAC run discarded."
}
```

### R2 - Noise +30 dB hypothesis, GNN 128/20

```json
{
  "after_changes": [
    "C06"
  ],
  "beamforming_K8_1W": {
    "GNN": 20.79,
    "OPT": 35.11,
    "MLP": 12.41,
    "ZF": 15.56
  },
  "deployment": "GNN-surface optimum (80, 138): 24 m from the paper's; rate map still flat."
}
```

### R3 - Sensitivity study of channel interpretations (OPT beamformer)

```json
{
  "after_changes": [
    "C07",
    "C08"
  ],
  "deployment": "Best: ln + horizontal + x-axis: optimum 17 m from the paper's, Fig. 8 profile error 0.15. See docs/history.json C08 for the full list."
}
```

### R4 - GNN capacity study on the calibrated model

```json
{
  "after_changes": [
    "C08",
    "C10"
  ],
  "deployment": "128/20: 45 m, 256/20: 8 m, 128/50: 9 m, 256/50: 0 m (GNN = 71 / 74 / 75 / 76% of OPT)."
}
```

### R5 - FINAL main profile (calibrated, GNN 256/50, SAC fixes)

```json
{
  "after_changes": [
    "C01-C13"
  ],
  "convergence": {
    "gnn_val": "10.25 +- 0.10",
    "mlp_val": "7.79 +- 0.11",
    "transfer_8_to_12": "12.52 (0 epochs to 95%) vs re-training 11.56 (51-63 epochs)"
  },
  "beamforming_K8_1W": {
    "GNN": 12.14,
    "OPT": 16.75,
    "MLP": 8.93,
    "ZF": 0.26,
    "MRT": 2.84
  },
  "gnn_over_opt": {
    "K=2": 0.72,
    "K=8": 0.72,
    "K=11": 0.7
  },
  "timing_seconds_per_inference_K8": {
    "GNN": 0.0024,
    "MLP": 0.0006,
    "OPT": 3.91
  },
  "deployment": {
    "optimum_gnn": [
      58,
      142
    ],
    "paper_optimum": [
      56,
      140
    ],
    "optimum_distance_to_paper_m": 2.8,
    "seed_optimum_distance_to_paper_m": [
      8.2,
      2.8,
      2.8
    ],
    "sac_final": [
      73.6,
      154.5
    ],
    "sac_to_optimum_m": 20.0,
    "sac_to_paper_optimum_m": 22.8,
    "rates_bps_hz": {
      "Graph-DRL": 19.63,
      "Area center": 14.09,
      "Geometric center": 15.81,
      "Circumcenter": 16.77,
      "Polygon centroid": 15.55,
      "Exhaustive search": 21.16
    },
    "graph_drl_vs_optimum_pct": 92.8,
    "graph_drl_gain_over_best_heuristic_pct": 17.0
  }
}
```

### R6 - FINAL reference profile (Table I as printed, GNN 256/50)

```json
{
  "after_changes": [
    "C01-C13 except C08"
  ],
  "convergence": {
    "gnn_val": "22.76 +- 3.01",
    "mlp_val": "20.91 +- 0.09",
    "transfer_8_to_12": "25.94 (never reaches 95%) vs re-training 28.60"
  },
  "beamforming_K8_1W": {
    "GNN": 25.66,
    "OPT": 90.85,
    "MLP": 21.76,
    "ZF": 89.96
  },
  "gnn_over_opt": {
    "K=2": 0.68,
    "K=8": 0.28,
    "K=11": 0.36
  },
  "deployment": {
    "optimum_gnn": [
      90,
      104
    ],
    "optimum_distance_to_paper_m": 49.5,
    "sac_final": [
      87.4,
      73.9
    ],
    "graph_drl_vs_optimum_pct": 93.1,
    "graph_drl_gain_over_best_heuristic_pct": 4.8
  }
}
```

## 5. Open issues

- GNN reaches only about 72% of our OPT on the calibrated model (paper: near-optimal). OPT is our own substitute for [45]; no direct comparison is possible.
- Fig. 8 heuristics reach 67-79% of the optimum here vs 39-61% in the paper (same ordering, smaller gaps).
- GNN at K=1 fails (no neighbours -> zero aggregate, never seen in training); MLP is better there.
- GNN trained at 1 W does not extrapolate in transmit power (saturates at 40 dBm); the paper's GNN tracks OPT.
- Table I reference profile: re-training beats transfer learning at K=12 (28.6 vs 25.9), unlike in the paper and in the calibrated profile.
- Eavesdropper-to-user pairing in the Fig. 7 layout is inferred; one cluster (U5/U6, E5/E6) is ambiguous.
- Only one SAC training seed per profile. The GNN-optimum robustness check covers 3 GNN seeds (2.8, 2.8 and 8.2 m from the paper's optimum in the main run).
