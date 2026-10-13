"""Fully connected baseline: flattened channels of all K pairs -> all K beamformers.

Unlike the GNN, the input/output size is tied to K, so a separate model is
needed for each number of users.
"""
from __future__ import annotations

import torch
from torch import nn

from .base import Beamformer, FeaturePReLU


class MLPBeamformer(Beamformer):
    KIND = "mlp"

    def __init__(self, num_antennas: int, ref_snr: float, num_users: int, hidden_dims: list[int],
                 input_transform: str = "linear") -> None:
        super().__init__(num_antennas, ref_snr, input_transform, num_users=num_users, hidden_dims=list(hidden_dims))
        self.num_users = num_users
        dims = [num_users * 4 * num_antennas, *hidden_dims]
        layers: list[nn.Module] = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(d_in, d_out), FeaturePReLU(d_out)]
        layers.append(nn.Linear(dims[-1], num_users * 2 * num_antennas))
        self.net = nn.Sequential(*layers)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        b, k, _ = x.shape
        if k != self.num_users:
            raise ValueError(f"MLP was built for K={self.num_users}, got K={k}")
        return self.net(x.reshape(b, -1)).reshape(b, k, 2 * self.num_antennas)
