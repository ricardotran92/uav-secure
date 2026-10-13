"""Optimisation-based reference beamforming ("OPT").

The paper compares against the precoding design of [45], which it does not
describe. As a reproducible stand-in we solve problem (10)-(11) per instance by
projected gradient ascent (Adam) on the sum secrecy rate, run from several
starting points (MRT, ZF and random) and keep the best iterate. This is a
strong local-optimisation baseline, not a certified global optimum.
"""
from __future__ import annotations

import torch

from ..config import OptConfig
from ..secrecy import normalize_power, project_power_ball, sum_secrecy_rate


def mrt(h: torch.Tensor, power: float) -> torch.Tensor:
    """Maximum-ratio transmission: w_k proportional to h_k."""
    return normalize_power(h, power)


def zero_forcing(h: torch.Tensor, power: float) -> torch.Tensor:
    """Zero-forcing toward legitimate users: h_k^H w_l = 0 for l != k (pseudo-inverse if K > N)."""
    a = h.conj()                                              # rows h_k^H, (B, K, N)
    w_cols = torch.linalg.pinv(a)                             # (B, N, K), a @ w_cols = I
    return normalize_power(w_cols.transpose(-1, -2), power)


@torch.enable_grad()
def optimize_beamforming(h: torch.Tensor, h_e: torch.Tensor, power: float, cfg: OptConfig,
                         gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the best beamformers (B, K, N) and their sum secrecy rates (B,).

    Gradients are enabled internally, so it can be called inside ``torch.no_grad()``.
    """
    b, k, n = h.shape
    r = max(cfg.restarts, 2)
    # Optimise in unit-power coordinates so the step size does not depend on P.
    inits = [mrt(h, 1.0), zero_forcing(h, 1.0)]
    noise = torch.complex(torch.randn((b, r - 2, k, n), generator=gen, device=h.device),
                          torch.randn((b, r - 2, k, n), generator=gen, device=h.device))
    v0 = torch.cat([torch.stack(inits, dim=1), noise], dim=1)  # (B, R, K, N)
    v = normalize_power(v0.reshape(b * r, k, n), 1.0).clone().requires_grad_(True)

    h_rep = h.repeat_interleave(r, dim=0)
    he_rep = h_e.repeat_interleave(r, dim=0)
    scale = power ** 0.5
    optimizer = torch.optim.Adam([v], lr=cfg.lr)

    best_rate = torch.full((b * r,), float("-inf"), device=h.device)
    best_w = torch.zeros_like(h_rep)
    for _ in range(cfg.iters + 1):
        w = scale * project_power_ball(v, 1.0)
        rate = sum_secrecy_rate(h_rep, he_rep, w)
        with torch.no_grad():
            improved = rate > best_rate
            best_rate = torch.where(improved, rate, best_rate)
            best_w[improved] = w[improved]
        optimizer.zero_grad(set_to_none=True)
        (-rate.sum()).backward()
        optimizer.step()

    best_rate, idx = best_rate.reshape(b, r).max(dim=1)
    best_w = best_w.reshape(b, r, k, n)[torch.arange(b, device=h.device), idx]
    return best_w.detach(), best_rate.detach()


def opt_beamformer(cfg: OptConfig, gen: torch.Generator):
    """(h, h_E, P) -> W using the OPT solver; same call signature as a trained network."""
    return lambda h, h_e, power: optimize_beamforming(h, h_e, power, cfg, gen)[0]
