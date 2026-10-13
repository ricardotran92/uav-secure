"""Phase 3 - beamforming benchmarks: Fig. 9a-c (sweeps), Fig. 10 (CDF), Fig. 11 (run time).

The GNN is the default-setting model from `exp_convergence.py` and is applied
to every scenario without retraining. The MLP has a K-dependent input size, so
one MLP per K is trained here with the same budget as the GNN.

Outputs (in outputs/runs/beamforming_<stamp>/): sweeps.csv, cdf.csv, timing.csv,
figs/fig9_sweeps.*, figs/fig10_cdf.*, figs/fig11_timing.*, metrics.json
"""
from __future__ import annotations

import argparse
import pandas as pd

from uavsec import plotting
from uavsec.config import add_config_args, config_from_args
from uavsec.evaluation.beamforming import PAPER_METHODS, BeamformingEvaluator
from uavsec.models import load_model, save_model
from uavsec.models.base import dbm_to_watt
from uavsec.training.beamformer_trainer import build_model, learning_rate, train_beamformer
from uavsec.utils import Run, find_checkpoint, resolve_device, seed_everything


def summarize(rates: dict, sweep: str, value: float) -> list[dict]:
    return [{"sweep": sweep, "value": value, "method": m, "rate_mean": r.mean().item(),
             "rate_std": r.std().item(), "num_samples": r.numel()} for m, r in rates.items()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--gnn", help="GNN checkpoint (default: latest convergence run)")
    args = parser.parse_args()
    cfg = config_from_args(args)
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    sys, ev = cfg.system, cfg.eval
    k0 = sys.num_users

    gnn_path = find_checkpoint(cfg, f"gnn_k{k0}.pt", args.gnn)
    if gnn_path is None:
        raise SystemExit("No GNN checkpoint found - run scripts/exp_convergence.py first or pass --gnn.")
    run = Run("beamforming", cfg)
    log = run.logger
    log.info("GNN checkpoint: %s", gnn_path)

    # One MLP per K (the default-K MLP from the convergence run is reused when present).
    mlps = {}
    for k in sorted(set(ev.sweep_users) | set(ev.cdf_users) | set(ev.timing_users) | {k0}):
        cached = find_checkpoint(cfg, f"mlp_k{k}.pt")
        if cached is not None:
            mlps[k] = load_model(cached, device)
            continue
        seed_everything(cfg.seed)
        model = build_model("mlp", cfg, num_users=k)
        train_beamformer(model, cfg, num_users=k, epochs=cfg.gnn.epochs, lr=learning_rate("mlp", cfg),
                         seed=cfg.seed, device=device, logger=log, tag=f"[mlp K={k}]")
        save_model(model, run.path("ckpt", f"mlp_k{k}.pt"))
        mlps[k] = model
    evaluator = BeamformingEvaluator(cfg, device, load_model(gnn_path, device), mlps)

    # ---- Fig. 9: sweeps over K, transmit power and noise power ----
    rows = []
    for k in ev.sweep_users:
        rows += summarize(evaluator.rates(*evaluator.instances(k), sys.power_max_w), "users", k)
        log.info("users K=%d done", k)
    h, h_e = evaluator.instances(k0)
    for p_dbm in ev.sweep_power_dbm:
        rows += summarize(evaluator.rates(h, h_e, dbm_to_watt(p_dbm)), "power", p_dbm)
        log.info("power %.0f dBm done", p_dbm)
    for noise in ev.sweep_noise_w:
        rows += summarize(evaluator.rates(*evaluator.instances(k0, noise_power_w=noise), sys.power_max_w),
                          "noise", noise)
        log.info("noise %.2e W done", noise)
    sweeps = pd.DataFrame(rows)
    sweeps.to_csv(run.path("sweeps.csv"), index=False)
    plotted = sweeps[sweeps["method"].isin(PAPER_METHODS)]
    plotting.save(plotting.fig9_sweeps({s: df for s, df in plotted.groupby("sweep")}), run.path("figs", "fig9_sweeps"))

    # ---- Fig. 10: per-instance distribution for the trained K and a larger, unseen K ----
    rows = []
    for k in ev.cdf_users:
        for method, rates in evaluator.rates(*evaluator.instances(k), sys.power_max_w).items():
            rows += [{"method": method, "num_users": k, "rate": float(r)} for r in rates]
    cdf = pd.DataFrame(rows)
    cdf.to_csv(run.path("cdf.csv"), index=False)
    plotting.save(plotting.fig10_cdf(cdf[cdf["method"].isin(PAPER_METHODS)]), run.path("figs", "fig10_cdf"))

    # ---- Fig. 11: computation time (networks: many runs, OPT: one run) ----
    rows = []
    for k in ev.timing_users:
        for method, seconds in evaluator.timing(k, ev.timing_runs).items():
            runs = 1 if method == "OPT" else ev.timing_runs
            rows.append({"method": method, "num_users": k, "seconds": seconds, "runs": runs,
                         "seconds_per_run": seconds / runs})
    timing = pd.DataFrame(rows)
    timing.to_csv(run.path("timing.csv"), index=False)
    plotting.save(plotting.fig11_timing(timing), run.path("figs", "fig11_timing"))

    # ---- Headline metrics ----
    default = sweeps[(sweeps["sweep"] == "users") & (sweeps["value"] == k0)].set_index("method")["rate_mean"]
    ratio = sweeps.pivot_table(index=["sweep", "value"], columns="method", values="rate_mean")
    ratio = (ratio["GNN"] / ratio["OPT"]).rename("gnn_over_opt").reset_index()
    cdf_stats = cdf.groupby(["num_users", "method"])["rate"].describe(percentiles=[0.1, 0.5, 0.9])
    run.finish({
        "gnn_checkpoint": str(gnn_path),
        "default_setting_rates": default.to_dict(),
        "gnn_over_opt_ratio": ratio.to_dict(orient="records"),
        "cdf_stats": {f"K={k}/{m}": row.to_dict() for (k, m), row in cdf_stats.iterrows()},
        "timing_seconds_per_run": timing.pivot_table(index="num_users", columns="method",
                                                     values="seconds_per_run").to_dict(),
    })


if __name__ == "__main__":
    main()
