"""Which unspecified modelling choices reproduce the paper's deployment results (Fig. 7 / Fig. 8)?

For every combination of the ``--vary`` options, the expected sum secrecy rate is
mapped over the area for the fixed Fig. 7 layout, and two things are compared with
the paper:
  * the location of the optimum vs the paper's "Optimal (through search)" marker;
  * the Fig. 8 profile: each heuristic's rate relative to the optimum.
The beamformer is model-free by default (OPT), so no network is retrained per variant.

Example:
  python scripts/study_deployment_sensitivity.py --set opt.restarts=4 opt.iters=150 \
      --vary system.distance_model=3d,horizontal system.array_axis=x,y \
             system.noise_power_w=1.2e-13,1.2e-10
"""
from __future__ import annotations

import argparse
import itertools
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from uavsec import plotting
from uavsec.baselines.deployment import exhaustive_search, expected_rate, heuristic_positions
from uavsec.baselines.opt_beamforming import mrt, opt_beamformer, zero_forcing
from uavsec.channel import load_layout
from uavsec.config import REPO_ROOT, Config, add_config_args, config_from_args
from uavsec.models import load_model
from uavsec.utils import Run, make_generator, resolve_device, seed_everything


def parse_vary(specs: list[str]) -> list[dict[str, dict[str, object]]]:
    """``section.key=v1,v2`` specs -> list of ``Config.replace`` kwargs (Cartesian product)."""
    axes = []
    for spec in specs:
        path, _, values = spec.partition("=")
        section, key = path.split(".")
        axes.append([(section, key, yaml.safe_load(v)) for v in values.split(",")])
    variants = []
    for combo in itertools.product(*axes):
        changes: dict[str, dict[str, object]] = {}
        for section, key, value in combo:
            changes.setdefault(section, {})[key] = value
        variants.append(changes)
    return variants


def make_beamformer(name: str, cfg: Config, device: torch.device):
    if name == "opt":
        return opt_beamformer(cfg.opt, make_generator(cfg.seed, device))
    if name == "mrt":
        return lambda h, h_e, p: mrt(h, p)
    if name == "zf":
        return lambda h, h_e, p: zero_forcing(h, p)
    return load_model(name, device)            # path to a trained GNN/MLP checkpoint


def label(changes: dict[str, dict[str, object]]) -> str:
    return ", ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}"
                     for section in changes.values() for k, v in section.items())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    parser.add_argument("--vary", nargs="+", required=True, metavar="SECTION.KEY=V1,V2")
    parser.add_argument("--beamformer", default="opt", help="opt | mrt | zf | path to a checkpoint")
    parser.add_argument("--grid-step", type=float, default=5.0)
    parser.add_argument("--fading", type=int, default=8)
    args = parser.parse_args()
    base = config_from_args(args)
    seed_everything(base.seed)
    device = resolve_device(base.device)
    run = Run("deployment_sensitivity", base)
    log = run.logger

    with open(REPO_ROOT / base.sac.layout_file, encoding="utf-8") as fh:
        reference = yaml.safe_load(fh)["paper_reference"]
    paper_opt = np.asarray(reference["optimum"])
    paper_rates = reference["fig8_rates"]
    paper_profile = {s: r / paper_rates["Graph-DRL"] for s, r in paper_rates.items() if s != "Graph-DRL"}

    rows, maps = [], []
    for changes in parse_vary(args.vary):
        cfg = base.replace(**changes)
        sys = cfg.system
        scenario = load_layout(cfg.sac.layout_file, sys, device)
        users = scenario.users[0].cpu().numpy()
        beamformer = make_beamformer(args.beamformer, cfg, device)
        search = exhaustive_search(scenario, beamformer, sys, args.grid_step, args.fading,
                                   seed=cfg.seed, refine_top=10)
        heur = heuristic_positions(users, sys.area_size)
        pos = torch.as_tensor(np.stack(list(heur.values())), dtype=torch.float32, device=device)
        rates, _ = expected_rate(pos, scenario, beamformer, sys, 8 * args.fading, seed=cfg.seed + 7)
        profile = {s: float(r) / search["optimum_rate"] for s, r in zip(heur, rates.cpu())}
        # With a flat rate surface the argmax is ill-defined, so also ask how good the
        # paper's optimum is in our model, and how large the near-optimal (>= 95%) region is.
        at_paper, _ = expected_rate(torch.as_tensor(paper_opt[None], dtype=torch.float32, device=device),
                                    scenario, beamformer, sys, 8 * args.fading, seed=cfg.seed + 7)
        row = {
            "variant": label(changes), **{k: v for sec in changes.values() for k, v in sec.items()},
            "opt_x": search["optimum"][0], "opt_y": search["optimum"][1],
            "distance_to_paper_opt_m": float(np.linalg.norm(search["optimum"] - paper_opt)),
            "optimum_rate": search["optimum_rate"],
            "paper_opt_relative_rate": float(at_paper) / search["optimum_rate"],
            "near_optimal_area_pct": float(100 * (search["rate"] >= 0.95 * search["rate"].max()).mean()),
            "fig8_profile_error": float(np.mean([abs(profile[s] - paper_profile[s]) for s in paper_profile])),
            **{f"ratio[{s}]": v for s, v in profile.items()},
        }
        rows.append(row)
        maps.append((row, search, scenario))
        log.info("%-70s optimum (%5.1f, %5.1f) dist %5.1f m | rate %6.2f | paper-opt %.0f%% of max | "
                 "Fig.8 profile error %.3f", row["variant"], *search["optimum"], row["distance_to_paper_opt_m"],
                 row["optimum_rate"], 100 * row["paper_opt_relative_rate"], row["fig8_profile_error"])

    table = pd.DataFrame(rows).sort_values(["distance_to_paper_opt_m", "fig8_profile_error"])
    table.to_csv(run.path("sensitivity.csv"), index=False)

    # Small multiples: relative rate map per variant, our optimum vs the paper's.
    cols = min(4, len(maps))
    nrows = math.ceil(len(maps) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(3.6 * cols, 3.9 * nrows), squeeze=False)
    for ax, (row, search, scenario) in zip(axes.flat, maps):
        rel = search["rate"] / search["rate"].max()
        ax.pcolormesh(search["grid_x"], search["grid_y"], rel, cmap="Blues", vmin=rel.min(), vmax=1, shading="auto")
        plotting.draw_nodes(ax, scenario.users[0].cpu().numpy(), scenario.eves[0].cpu().numpy())
        ax.scatter(*search["optimum"], marker="*", s=160, color=plotting.PALETTE[1], edgecolor="white", zorder=5)
        ax.scatter(*paper_opt, marker="X", s=90, color=plotting.PALETTE[6], edgecolor="white", zorder=5)
        ax.set(xlim=(0, base.system.area_size), ylim=(0, base.system.area_size), aspect="equal",
               xticks=[], yticks=[])
        ax.set_title(f"{row['variant']}\n{row['distance_to_paper_opt_m']:.0f} m off, "
                     f"min/max {rel.min():.2f}", fontsize=7)
    for ax in axes.flat[len(maps):]:
        ax.set_visible(False)
    fig.suptitle(f"Relative expected secrecy-rate maps ({args.beamformer}); "
                 "star = our optimum, X = paper optimum", fontsize=10)
    fig.tight_layout()
    plotting.save(fig, run.path("figs", "sensitivity_maps"))

    run.finish({"beamformer": args.beamformer, "paper_optimum": paper_opt, "paper_fig8_profile": paper_profile,
                "variants": table.to_dict(orient="records")})


if __name__ == "__main__":
    main()
