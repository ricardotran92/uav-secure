"""Beamforming benchmarks at a fixed UAV position: GNN vs OPT vs MLP (+ ZF/MRT references).

All methods see exactly the same test instances: the instance generator is
re-seeded per sweep point, and a noise-power change only rescales the
(noise-normalised) channels, so every curve compares like with like.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..baselines.opt_beamforming import mrt, optimize_beamforming, zero_forcing
from ..channel import area_center, channels, sample_scenario, sample_uav_positions
from ..config import Config
from ..models import Beamformer
from ..secrecy import sum_secrecy_rate
from ..utils import make_generator, timer

PAPER_METHODS = ("GNN", "OPT", "MLP")       # methods shown in the paper's figures
REFERENCE_METHODS = ("ZF", "MRT")            # closed-form references, reported in tables


@dataclass
class BeamformingEvaluator:
    cfg: Config
    device: torch.device
    gnn: Beamformer
    mlps: dict[int, Beamformer] = field(default_factory=dict)   # one MLP per K
    seed: int = 12345

    def instances(self, num_users: int, noise_power_w: float | None = None,
                  num: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        sys, ev = self.cfg.system, self.cfg.eval
        n = num or ev.test_samples
        gen = make_generator(self.seed + num_users, self.device)
        scenario = sample_scenario(sys, n, gen, self.device, num_users=num_users)
        if ev.uav_position == "center":
            uav = area_center(sys, n, self.device)
        elif ev.uav_position == "random":
            uav = sample_uav_positions(sys, n, gen, self.device)
        else:
            raise ValueError(f"eval.uav_position must be 'center' or 'random', got {ev.uav_position!r}")
        return channels(sys, uav, scenario, gen, noise_power_w)

    @torch.no_grad()
    def _network_rate(self, model: Beamformer, h, h_e, power) -> torch.Tensor:
        return sum_secrecy_rate(h, h_e, model(h, h_e, power))

    def rates(self, h: torch.Tensor, h_e: torch.Tensor, power: float) -> dict[str, torch.Tensor]:
        """Sum secrecy rate per instance for every available method."""
        k = h.shape[1]
        out = {"GNN": self._network_rate(self.gnn, h, h_e, power)}
        _, out["OPT"] = optimize_beamforming(h, h_e, power, self.cfg.opt, make_generator(self.seed, self.device))
        if k in self.mlps:
            out["MLP"] = self._network_rate(self.mlps[k], h, h_e, power)
        out["ZF"] = sum_secrecy_rate(h, h_e, zero_forcing(h, power))
        out["MRT"] = sum_secrecy_rate(h, h_e, mrt(h, power))
        return {name: r.cpu() for name, r in out.items()}

    def timing(self, num_users: int, runs: int) -> dict[str, float]:
        """Wall time of ``runs`` single-instance inferences (GNN/MLP) vs one OPT solve."""
        h, h_e = self.instances(num_users, num=1)
        power = self.cfg.system.power_max_w
        result = {}
        networks = {"GNN": self.gnn, **({"MLP": self.mlps[num_users]} if num_users in self.mlps else {})}
        with torch.no_grad():
            for name, model in networks.items():
                model(h, h_e, power)                            # warm-up (CUDA kernels, caches)
                with timer() as t:
                    for _ in range(runs):
                        model(h, h_e, power)
                result[name] = t["seconds"]
        with timer() as t:
            optimize_beamforming(h, h_e, power, self.cfg.opt, make_generator(self.seed, self.device))
        result["OPT"] = t["seconds"]
        return result
