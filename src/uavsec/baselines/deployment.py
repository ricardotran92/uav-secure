"""UAV deployment baselines (Fig. 7-8): node-location heuristics and exhaustive grid search.

A position is scored by its *expected* sum secrecy rate: the mean over many
small-scale fading draws, with the GNN computing the beamformers - the same
inner layer the SAC agent uses for its reward.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from scipy.spatial import ConvexHull, QhullError

from ..channel import Scenario, channels
from ..config import SystemConfig
from ..secrecy import sum_secrecy_rate
from ..utils import make_generator

# (h, h_E, P) -> beamformers W; satisfied by trained networks and by the OPT solver.
BeamformFn = Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]


def min_enclosing_circle_center(points: np.ndarray) -> np.ndarray:
    """Centre of the smallest circle containing all points (Welzl's algorithm, iterative)."""
    pts = np.random.default_rng(0).permutation(points)

    def circle_two(a, b):
        c = (a + b) / 2
        return c, np.linalg.norm(a - c)

    def circle_three(a, b, c):
        d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
        if abs(d) < 1e-12:                        # collinear: use the farthest pair
            pairs = [(a, b), (a, c), (b, c)]
            return max((circle_two(p, q) for p, q in pairs), key=lambda cr: cr[1])
        sq = [p @ p for p in (a, b, c)]
        ux = (sq[0] * (b[1] - c[1]) + sq[1] * (c[1] - a[1]) + sq[2] * (a[1] - b[1])) / d
        uy = (sq[0] * (c[0] - b[0]) + sq[1] * (a[0] - c[0]) + sq[2] * (b[0] - a[0])) / d
        center = np.array([ux, uy])
        return center, np.linalg.norm(a - center)

    def inside(circle, p):
        return np.linalg.norm(p - circle[0]) <= circle[1] + 1e-9

    circle = (pts[0], 0.0)
    for i in range(1, len(pts)):
        if inside(circle, pts[i]):
            continue
        circle = (pts[i], 0.0)
        for j in range(i):
            if inside(circle, pts[j]):
                continue
            circle = circle_two(pts[i], pts[j])
            for m in range(j):
                if not inside(circle, pts[m]):
                    circle = circle_three(pts[i], pts[j], pts[m])
    return circle[0]


def polygon_centroid(points: np.ndarray) -> np.ndarray:
    """Area centroid of the convex hull of the points (falls back to the mean if degenerate)."""
    try:
        hull = points[ConvexHull(points).vertices]
    except (QhullError, ValueError):
        return points.mean(axis=0)
    x, y = hull[:, 0], hull[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area = cross.sum() / 2
    if abs(area) < 1e-12:
        return points.mean(axis=0)
    cx = ((x + np.roll(x, -1)) * cross).sum() / (6 * area)
    cy = ((y + np.roll(y, -1)) * cross).sum() / (6 * area)
    return np.array([cx, cy])


def heuristic_positions(users: np.ndarray, area_size: float) -> dict[str, np.ndarray]:
    """The four heuristic deployments of Fig. 8, computed from legitimate-user locations."""
    return {
        "Area center": np.array([area_size / 2, area_size / 2]),
        "Geometric center": users.mean(axis=0),
        "Circumcenter": min_enclosing_circle_center(users),
        "Polygon centroid": polygon_centroid(users),
    }


@torch.no_grad()
def expected_rate(positions: torch.Tensor, scenario: Scenario, model: BeamformFn, sys: SystemConfig,
                  fading_samples: int, seed: int, chunk: int = 8192) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and std (over fading) of the sum secrecy rate for UAV ``positions`` (P, 2).

    ``model`` maps (h, h_E, P) to beamformers: a trained network or e.g. the OPT solver.
    ``scenario`` is a single instance. The fading generator is re-seeded on every
    call, so different strategies are compared under common random numbers.
    """
    gen = make_generator(seed, positions.device)
    uav = positions.repeat_interleave(fading_samples, dim=0)            # (P*M, 2)
    rates = []
    for i in range(0, uav.shape[0], chunk):
        part = uav[i:i + chunk]
        h, h_e = channels(sys, part, scenario.expand(part.shape[0]), gen)
        rates.append(sum_secrecy_rate(h, h_e, model(h, h_e, sys.power_max_w)))
    rates = torch.cat(rates).reshape(-1, fading_samples)
    return rates.mean(dim=1), rates.std(dim=1)


def exhaustive_search(scenario: Scenario, model: BeamformFn, sys: SystemConfig, step_m: float,
                      fading_samples: int, seed: int, refine_top: int) -> dict[str, np.ndarray]:
    """Grid search of the expected secrecy rate over the whole area.

    The coarse map uses ``fading_samples`` draws per point; the ``refine_top``
    best points are then re-scored with 8x more draws, so the reported optimum
    is not just the luckiest Monte-Carlo estimate.
    """
    device = scenario.users.device
    axis = torch.arange(0.0, sys.area_size + 1e-6, step_m, device=device)
    gx, gy = torch.meshgrid(axis, axis, indexing="xy")
    grid = torch.stack((gx.flatten(), gy.flatten()), dim=-1)
    mean, _ = expected_rate(grid, scenario, model, sys, fading_samples, seed)
    top = mean.topk(min(refine_top, mean.numel())).indices
    refined, _ = expected_rate(grid[top], scenario, model, sys, 8 * fading_samples, seed + 1)
    best = int(top[refined.argmax()])
    return {
        "grid_x": gx.cpu().numpy(), "grid_y": gy.cpu().numpy(),
        "rate": mean.reshape(gx.shape).cpu().numpy(),
        "optimum": grid[best].cpu().numpy(), "optimum_rate": float(refined.max()),
    }
