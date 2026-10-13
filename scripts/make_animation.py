"""Phase 5 - animate the learned UAV trajectory of the latest deployment run (GIF, and MP4 if ffmpeg exists)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from uavsec.animation import animate_trajectory, replay
from uavsec.channel import Scenario
from uavsec.config import add_config_args, config_from_args
from uavsec.models import load_model
from uavsec.utils import latest_run, load_json, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--run", help="deployment run directory (default: latest)")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    cfg = config_from_args(args)
    device = resolve_device(cfg.device)

    run_dir = Path(args.run) if args.run else latest_run(cfg.output_path, "deployment")
    if run_dir is None:
        raise SystemExit("No deployment run found - run scripts/exp_deployment.py first.")
    scen = load_json(run_dir / "scenario.json")
    users, eves = np.asarray(scen["users"]), np.asarray(scen["eves"])
    scenario = Scenario(torch.as_tensor(users, dtype=torch.float32, device=device).unsqueeze(0),
                        torch.as_tensor(eves, dtype=torch.float32, device=device).unsqueeze(0))
    model = load_model(Path(scen["gnn_checkpoint"]), device)
    positions = pd.read_csv(run_dir / "trajectory.csv")[["x", "y"]].to_numpy()

    beam_power, secrecy = replay(positions, scenario, model, cfg.system, seed=cfg.seed)
    outputs = animate_trajectory(positions, users, eves, beam_power, secrecy, cfg.system.area_size,
                                 cfg.system.power_max_w, run_dir / "animation" / "uav_trajectory", fps=args.fps)
    print("written:", *map(str, outputs), sep="\n  ")


if __name__ == "__main__":
    main()
