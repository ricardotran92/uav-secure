"""Phase 2 - GNN security beamforming: Fig. 4 (GNN vs MLP) and Fig. 5 (transfer 8 -> 12 users).

Outputs (in outputs/runs/convergence_<stamp>/):
  convergence.csv, transfer.csv      per-epoch training curves
  ckpt/gnn_k{K}.pt, ckpt/mlp_k{K}.pt best-seed models used by later phases
  figs/fig4_convergence.*, figs/fig5_transfer.*
  metrics.json
"""
from __future__ import annotations

import argparse
import shutil

import numpy as np
import pandas as pd

from uavsec import plotting
from uavsec.config import add_config_args, config_from_args
from uavsec.models import count_parameters, load_model, save_model
from uavsec.training.beamformer_trainer import TrainHistory, build_model, learning_rate, train_beamformer
from uavsec.utils import Run, resolve_device, seed_everything


def curve_rows(history: TrainHistory, **labels) -> list[dict]:
    return [{**labels, "epoch": e, "train_loss": loss, "val_rate": val}
            for e, (loss, val) in enumerate(zip(history.train_loss, history.val_rate))]


def mean_std(values: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def first_epoch_reaching(curve: list[float], target: float) -> int | None:
    """First epoch whose validation rate reaches ``target`` (None if never)."""
    hits = np.nonzero(np.asarray(curve) >= target)[0]
    return int(hits[0]) if hits.size else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    cfg = config_from_args(parser.parse_args())
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    run = Run("convergence", cfg)
    log = run.logger
    k = cfg.system.num_users
    seeds = cfg.eval.convergence_seeds

    # ---- Fig. 4: GNN vs MLP at the default K, several seeds ----
    rows, summary, best_ckpt = [], {}, {}
    for kind in ("gnn", "mlp"):
        results = []
        for seed in seeds:
            seed_everything(seed)
            model = build_model(kind, cfg)
            hist = train_beamformer(model, cfg, num_users=k, epochs=cfg.gnn.epochs, lr=learning_rate(kind, cfg),
                                    seed=seed, device=device, logger=log, tag=f"[{kind} K={k} seed={seed}]")
            ckpt = run.path("ckpt", f"{kind}_k{k}_seed{seed}.pt")
            save_model(model, ckpt, history=hist.__dict__)
            rows += curve_rows(hist, model=kind.upper(), seed=seed)
            results.append((hist, ckpt))
        best_hist, best_path = max(results, key=lambda r: r[0].best_val_rate)
        best_ckpt[kind] = run.path("ckpt", f"{kind}_k{k}.pt")
        shutil.copyfile(best_path, best_ckpt[kind])
        tail = max(1, cfg.gnn.epochs // 30)          # average the last ~3% of epochs
        summary[kind.upper()] = {
            "final_loss": mean_std([np.mean(h.train_loss[-tail:]) for h, _ in results]),
            "best_val_sum_secrecy_rate": mean_std([h.best_val_rate for h, _ in results]),
            "train_time_s": mean_std([h.train_time_s for h, _ in results]),
            "num_parameters": count_parameters(build_model(kind, cfg)),
            "best_seed_checkpoint": best_ckpt[kind].relative_to(run.dir).as_posix(),
        }
        log.info("%s summary: %s", kind.upper(), summary[kind.upper()])

    conv = pd.DataFrame(rows)
    conv.to_csv(run.path("convergence.csv"), index=False)
    plotting.save(plotting.fig4_convergence(conv), run.path("figs", "fig4_convergence"))

    # ---- Fig. 5: transfer learning (init from K-user GNN) vs re-training for K' users ----
    k2, epochs2 = cfg.eval.transfer_users, cfg.eval.transfer_epochs
    rows, finals = [], {"Transfer learning": [], "Re-training": []}
    curves = {"Transfer learning": [], "Re-training": []}
    for seed in seeds:
        for method in finals:
            seed_everything(seed)
            model = load_model(best_ckpt["gnn"], device) if method == "Transfer learning" else build_model("gnn", cfg)
            hist = train_beamformer(model, cfg, num_users=k2, epochs=epochs2, lr=cfg.gnn.lr, seed=seed,
                                    device=device, logger=log, tag=f"[{method} K={k2} seed={seed}]")
            rows += curve_rows(hist, method=method, seed=seed, num_users=k2)
            finals[method].append(hist.best_val_rate)
            curves[method].append(hist.val_rate)
            if method == "Transfer learning" and seed == seeds[0]:
                save_model(model, run.path("ckpt", f"gnn_k{k2}_transfer.pt"))
    transfer = pd.DataFrame(rows)
    transfer.to_csv(run.path("transfer.csv"), index=False)
    plotting.save(plotting.fig5_transfer(transfer), run.path("figs", "fig5_transfer"))

    # Convergence speed: epochs needed to reach 95% of the re-trained model's best rate.
    target = 0.95 * float(np.mean(finals["Re-training"]))
    transfer_summary = {
        method: {
            "best_val_sum_secrecy_rate": mean_std(vals),
            "epochs_to_95pct_of_retrain": [first_epoch_reaching(c, target) for c in curves[method]],
            "initial_val_rate": mean_std([c[0] for c in curves[method]]),
        }
        for method, vals in finals.items()
    }
    run.finish({"num_users": k, "transfer_users": k2, "beamforming": summary, "transfer": transfer_summary})


if __name__ == "__main__":
    main()
