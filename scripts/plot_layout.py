"""Draw the ground-node layout used by the SAC experiment, to check it before any training run.

Each user is linked to its eavesdropper by a dashed line labelled with their distance.
Usage: python scripts/plot_layout.py [--out layout.png] [--set sac.layout_file=...]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from uavsec.channel import load_layout
from uavsec.config import REPO_ROOT, add_config_args, config_from_args
from uavsec.plotting import EVE_COLOR, INK, MUTED, PALETTE, USER_COLOR, draw_nodes, save


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--out", default=str(REPO_ROOT / "outputs" / "layout" / "layout_check"))
    cfg = config_from_args(args := parser.parse_args())
    if not cfg.sac.layout_file:
        raise SystemExit("sac.layout_file is empty (random layout); nothing fixed to plot.")
    sys = cfg.system
    scenario = load_layout(cfg.sac.layout_file, sys, torch.device("cpu"))
    users, eves = scenario.users[0].numpy(), scenario.eves[0].numpy()
    dist = np.linalg.norm(users - eves, axis=1)

    fig, ax = plt.subplots(figsize=(7, 7))
    draw_nodes(ax, users, eves)
    for k, (u, e, d) in enumerate(zip(users, eves, dist)):
        ax.plot(*zip(u, e), ls="--", color=MUTED, lw=0.9, zorder=1)
        ax.annotate(f"U{k}", u, xytext=(5, 5), textcoords="offset points", color=USER_COLOR, fontsize=9, weight="bold")
        ax.annotate(f"E{k}", e, xytext=(5, -12), textcoords="offset points", color=EVE_COLOR, fontsize=9, weight="bold")
        mid = (u + e) / 2
        ax.annotate(f"{d:.0f} m", mid, xytext=(-4, 4), textcoords="offset points", color=MUTED, fontsize=7, ha="right")
    start = np.full(2, sys.area_size / 2)
    ax.scatter(*start, marker="^", s=110, color=PALETTE[6], label="UAV start (area center)", zorder=4)
    ax.set(xlim=(0, sys.area_size), ylim=(0, sys.area_size), aspect="equal", xlabel="x (m)", ylabel="y (m)",
           title=f"SAC layout: {Path(cfg.sac.layout_file).name}  (K={len(users)}, user-eve distance "
                 f"{dist.min():.0f}-{dist.max():.0f} m, mean {dist.mean():.0f} m)")
    ax.set_xticks(range(0, int(sys.area_size) + 1, 20))
    ax.set_yticks(range(0, int(sys.area_size) + 1, 20))
    ax.grid(True, color="#e4e3df", lw=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, fontsize=9)
    for path in save(fig, Path(args.out)):
        print("written:", path)
    print("user-eve distances (m):", np.round(dist, 1).tolist())


if __name__ == "__main__":
    main()
