"""Phase 4 - SAC-based UAV deployment: Fig. 6 (SAC convergence), Fig. 7 (trajectory vs
exhaustive optimum) and Fig. 8 (comparison with heuristic deployments).

Outputs (in outputs/runs/deployment_<stamp>/): episodes.csv, trajectory.csv,
deployment.csv, scenario.json, rate_map.npz, sac_model.zip, figs/fig6..8.*, metrics.json
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from uavsec import plotting
from uavsec.baselines.deployment import exhaustive_search, expected_rate, heuristic_positions
from uavsec.baselines.opt_beamforming import opt_beamformer
from uavsec.channel import load_layout, load_paper_reference, sample_scenario
from uavsec.config import OptConfig, add_config_args, config_from_args
from uavsec.envs.uav_env import UAVDeploymentEnv
from uavsec.models import load_model
from uavsec.rl.sac_trainer import rollout, train_sac
from uavsec.utils import Run, find_checkpoint, make_generator, resolve_device, save_json, seed_everything


def convergence_episode(values: np.ndarray, window: int = 20, frac: float = 0.95) -> int | None:
    """First episode where the moving average reaches ``frac`` of its final value."""
    ma = pd.Series(values).rolling(window, min_periods=1).mean().to_numpy()
    hits = np.nonzero(ma >= frac * ma[-1])[0]
    return int(hits[0]) if hits.size else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--gnn", help="GNN checkpoint (default: latest convergence run)")
    args = parser.parse_args()
    cfg = config_from_args(args)
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    sys, ev = cfg.system, cfg.eval

    gnn_path = find_checkpoint(cfg, f"gnn_k{sys.num_users}.pt", args.gnn)
    if gnn_path is None:
        raise SystemExit("No GNN checkpoint found - run scripts/exp_convergence.py first or pass --gnn.")
    run = Run("deployment", cfg)
    log = run.logger
    gnn = load_model(gnn_path, device)
    scenario = (load_layout(cfg.sac.layout_file, sys, device) if cfg.sac.layout_file
                else sample_scenario(sys, 1, make_generator(cfg.sac.topology_seed, device), device))
    users, eves = scenario.users[0].cpu().numpy(), scenario.eves[0].cpu().numpy()
    save_json({"users": users, "eves": eves, "gnn_checkpoint": str(gnn_path)}, run.path("scenario.json"))

    # ---- Fig. 6: SAC training ----
    env = UAVDeploymentEnv(cfg, gnn, scenario, device, seed=cfg.seed)
    agent, episodes = train_sac(env, cfg.sac, device, cfg.seed, log)
    agent.save(run.path("sac_model.zip"))
    episodes.to_csv(run.path("episodes.csv"), index=False)
    plotting.save(plotting.fig6_sac(episodes), run.path("figs", "fig6_sac"))

    # ---- Fig. 7: learned trajectory vs exhaustive search ----
    traj = rollout(agent, env, seed=cfg.seed + 1)
    traj.to_csv(run.path("trajectory.csv"), index=False)
    # The optimum of problem (7)-(9) optimises W and q jointly, so by default the search
    # uses the OPT solver; the GNN-based optimum (what the SAC reward sees) is reported too.
    search_opt = OptConfig(restarts=ev.search_opt_restarts, iters=ev.search_opt_iters, lr=cfg.opt.lr)
    beamformers = {"opt": opt_beamformer(search_opt, make_generator(cfg.seed, device)), "gnn": gnn}
    if ev.search_beamformer not in beamformers:
        raise ValueError(f"eval.search_beamformer must be one of {sorted(beamformers)}")
    search = exhaustive_search(scenario, beamformers[ev.search_beamformer], sys, ev.grid_step_m, ev.fading_samples,
                               seed=cfg.seed, refine_top=ev.search_refine_top)
    gnn_search = search if ev.search_beamformer == "gnn" else exhaustive_search(
        scenario, gnn, sys, ev.grid_step_m, ev.fading_samples, seed=cfg.seed, refine_top=ev.search_refine_top)
    log.info("optimum (%s): (%.1f, %.1f) | GNN-surface optimum: (%.1f, %.1f)", ev.search_beamformer,
             *search["optimum"], *gnn_search["optimum"])
    np.savez_compressed(run.path("rate_map.npz"), **{k: v for k, v in search.items() if isinstance(v, np.ndarray)})
    final = traj[["x", "y"]].to_numpy()[-1]
    reference = load_paper_reference(cfg.sac.layout_file) if cfg.sac.layout_file else None
    paper_opt = np.asarray(reference["optimum"]) if reference else None
    plotting.save(plotting.fig7_trajectory(users, eves, traj[["x", "y"]].to_numpy(), search["optimum"], sys.area_size,
                                           (search["grid_x"], search["grid_y"], search["rate"]), paper_optimum=paper_opt),
                  run.path("figs", "fig7_trajectory"))

    # ---- Fig. 8: all deployment strategies, scored with the deployed (GNN) beamformer ----
    # "Exhaustive search" is the best position for GNN beamforming, i.e. the upper bound for Graph-DRL.
    strategies = {"Graph-DRL": final, **heuristic_positions(users, sys.area_size),
                  "Exhaustive search": gnn_search["optimum"]}
    pos = torch.as_tensor(np.stack(list(strategies.values())), dtype=torch.float32, device=device)
    mean, std = expected_rate(pos, scenario, gnn, sys, ev.fading_samples * 8, seed=cfg.seed + 7)
    deployment = pd.DataFrame({"strategy": list(strategies), "x": pos[:, 0].cpu(), "y": pos[:, 1].cpu(),
                               "rate": mean.cpu(), "rate_std": std.cpu()})
    deployment.to_csv(run.path("deployment.csv"), index=False)
    plotting.save(plotting.fig8_deployment(deployment, users, eves, sys.area_size), run.path("figs", "fig8_deployment"))

    rates = deployment.set_index("strategy")["rate"]
    paper_metrics = {}
    if paper_opt is not None:
        paper_metrics = {
            "paper_optimum_position": paper_opt,
            "optimum_distance_to_paper_m": float(np.linalg.norm(search["optimum"] - paper_opt)),
            "sac_final_distance_to_paper_optimum_m": float(np.linalg.norm(final - paper_opt)),
        }
    run.finish({
        "gnn_checkpoint": str(gnn_path),
        "search_beamformer": ev.search_beamformer,
        "final_position": final, "optimum_position": search["optimum"],
        "gnn_surface_optimum_position": gnn_search["optimum"],
        "distance_to_optimum_m": float(np.linalg.norm(final - search["optimum"])),
        "strategy_rates": rates.to_dict(),
        "graph_drl_vs_optimum_pct": float(100 * rates["Graph-DRL"] / rates["Exhaustive search"]),
        "graph_drl_gain_over_best_heuristic_pct": float(
            100 * (rates["Graph-DRL"] / rates.drop(["Graph-DRL", "Exhaustive search"]).max() - 1)),
        "sac_convergence_episode": convergence_episode(episodes["episode_reward"].to_numpy()),
        "sac_final_mean_secrecy_rate": float(episodes["mean_secrecy_rate"].tail(20).mean()),
        "trajectory_steps": len(traj) - 1,
        **paper_metrics,
    })


if __name__ == "__main__":
    main()
