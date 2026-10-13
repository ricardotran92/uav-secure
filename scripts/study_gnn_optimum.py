"""How stable is the GNN-based deployment optimum (Fig. 7) across training seeds?

For each seed a GNN is trained from scratch with the given config, the expected
secrecy-rate map of the Fig. 7 layout is searched with that GNN (problem (23)-(25)),
and the optimum is compared with the paper's.

Example:
  python scripts/study_gnn_optimum.py --config configs/default.yaml configs/calibrated.yaml \
      --set gnn.hidden_dim=256 --seeds 0 1 2
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from uavsec.baselines.deployment import exhaustive_search, expected_rate
from uavsec.channel import load_layout, load_paper_reference
from uavsec.config import add_config_args, config_from_args
from uavsec.models import save_model
from uavsec.training.beamformer_trainer import build_model, train_beamformer
from uavsec.utils import Run, resolve_device, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    cfg = config_from_args(args)
    device = resolve_device(cfg.device)
    sys, ev = cfg.system, cfg.eval
    run = Run("gnn_optimum", cfg)
    log = run.logger

    scenario = load_layout(cfg.sac.layout_file, sys, device)
    paper_opt = np.asarray(load_paper_reference(cfg.sac.layout_file)["optimum"])
    paper_pos = torch.as_tensor(paper_opt[None], dtype=torch.float32, device=device)

    rows = []
    for seed in args.seeds:
        seed_everything(seed)
        model = build_model("gnn", cfg)
        hist = train_beamformer(model, cfg, num_users=sys.num_users, epochs=cfg.gnn.epochs, lr=cfg.gnn.lr,
                                seed=seed, device=device, logger=log, tag=f"[seed {seed}]")
        save_model(model, run.path("ckpt", f"gnn_seed{seed}.pt"))
        search = exhaustive_search(scenario, model, sys, ev.grid_step_m, ev.fading_samples, seed=cfg.seed,
                                   refine_top=ev.search_refine_top)
        at_paper, _ = expected_rate(paper_pos, scenario, model, sys, 8 * ev.fading_samples, seed=cfg.seed + 1)
        row = {"seed": seed, "best_val_rate": hist.best_val_rate, "opt_x": search["optimum"][0],
               "opt_y": search["optimum"][1], "optimum_rate": search["optimum_rate"],
               "distance_to_paper_opt_m": float(np.linalg.norm(search["optimum"] - paper_opt)),
               "paper_opt_relative_rate": float(at_paper) / search["optimum_rate"]}
        rows.append(row)
        log.info("seed %d | val %.2f | optimum (%.0f, %.0f) | %.1f m from paper | paper point %.0f%% of max",
                 seed, row["best_val_rate"], row["opt_x"], row["opt_y"], row["distance_to_paper_opt_m"],
                 100 * row["paper_opt_relative_rate"])

    table = pd.DataFrame(rows)
    table.to_csv(run.path("gnn_optimum.csv"), index=False)
    dist = table["distance_to_paper_opt_m"]
    run.finish({"paper_optimum": paper_opt, "seeds": table.to_dict(orient="records"),
                "distance_to_paper_opt_m": {"mean": dist.mean(), "max": dist.max()},
                "within_20m": int((dist <= 20).sum())})


if __name__ == "__main__":
    main()
