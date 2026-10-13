"""Achievable, wiretap and secrecy rates (eqs. (4)-(6), (19), (21)-(22)).

Channels are noise-normalised (see ``channel.py``), so the noise power is 1.
Beamformers ``w`` are complex (B, K, N) with row k = w_k. Rates are in bps/Hz.

PyTorch differentiates complex tensors natively, so the real-domain rewrite of
eq. (21)-(22) is not needed for training; ``tests/test_secrecy.py`` checks that
both forms agree.
"""
from __future__ import annotations

import torch


def link_gains(h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """G[b, k, l] = |h_k^H w_l|^2, shape (B, K, K)."""
    return torch.einsum("bkn,bln->bkl", h.conj(), w).abs().square()


def sinr_rates(h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """log2(1 + |h_k^H w_k|^2 / (sum_{l != k} |h_k^H w_l|^2 + 1)), shape (B, K)."""
    gains = link_gains(h, w)
    signal = gains.diagonal(dim1=1, dim2=2)
    interference = gains.sum(-1) - signal
    return torch.log2(1.0 + signal / (interference + 1.0))


def secrecy_rates(h: torch.Tensor, h_e: torch.Tensor, w: torch.Tensor,
                  clamp: bool = True) -> torch.Tensor:
    """Per-user secrecy rate R_k^sec = (R_k - R_E,k)^+, shape (B, K)."""
    diff = sinr_rates(h, w) - sinr_rates(h_e, w)
    return diff.clamp_min(0.0) if clamp else diff


def sum_secrecy_rate(h: torch.Tensor, h_e: torch.Tensor, w: torch.Tensor,
                     clamp: bool = True) -> torch.Tensor:
    """System secrecy rate sum_k R_k^sec, shape (B,)."""
    return secrecy_rates(h, h_e, w, clamp).sum(-1)


def normalize_power(raw: torch.Tensor, power: float | torch.Tensor) -> torch.Tensor:
    """Eq. (19): scale (B, K, N) beamformers so that sum_k ||w_k||^2 = P.

    (The paper writes P^max in front of the ratio; sqrt(P^max) is what meets
    constraint (11) with equality, which is used here.)
    """
    norm = raw.abs().square().sum(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-12)
    power = torch.as_tensor(power, dtype=norm.dtype, device=raw.device)
    return torch.sqrt(power).reshape(-1, 1, 1) * raw / norm


def project_power_ball(raw: torch.Tensor, power: float | torch.Tensor) -> torch.Tensor:
    """Projection onto sum_k ||w_k||^2 <= P (scales down only when infeasible)."""
    norm = raw.abs().square().sum(dim=(1, 2), keepdim=True).sqrt()
    power = torch.as_tensor(power, dtype=norm.dtype, device=raw.device).reshape(-1, 1, 1)
    scale = torch.clamp(torch.sqrt(power) / norm.clamp_min(1e-12), max=1.0)
    return raw * scale
