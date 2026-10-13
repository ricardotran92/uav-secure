"""GNN for security beamforming (Section IV-A, Fig. 2).

Graph: one node per legitimate user; node k carries the channels of user k and
its eavesdropper plus a learned embedding e_k in C^N (eq. (12)). Edges are
implicit (fully connected) - interference is learned through message passing.

Each GCN layer d performs
    z_k   = [Re h_k, Im h_k, Re h_E,k, Im h_E,k, Re e_k, Im e_k]      (13)
    m_k   = f_gen(z_k)                                                (14)
    a_k   = max_{l != k} f_aggr(m_l)                                  (16)
    c_k   = [m_k, a_k]                                                (17)
    e_k   = f_out(c_k)                                                (18)
and the final embeddings are mapped to beamformers by eq. (19).

The f_* networks are shared by all nodes, which makes the model permutation
equivariant and independent of K (needed for transfer/generalisation).
"""
from __future__ import annotations

import torch
from torch import nn

from .base import Beamformer, FeaturePReLU


def max_over_others(x: torch.Tensor) -> torch.Tensor:
    """a_k = max_{l != k} x_l along the node axis. x: (B, K, H) -> (B, K, H).

    For K = 1 there are no neighbours and the aggregate is zero.
    """
    k = x.shape[1]
    if k == 1:
        return torch.zeros_like(x)
    # Top-2 trick: the max over the others is the overall max, except for the node that
    # holds it, which gets the second largest. O(K) memory instead of O(K^2).
    top, idx = x.topk(2, dim=1)                                 # (B, 2, H)
    node = torch.arange(k, device=x.device).reshape(1, k, 1)
    return torch.where(node == idx[:, :1], top[:, 1:2], top[:, :1])


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.f_gen = nn.Sequential(nn.Linear(in_dim, hidden_dim), FeaturePReLU(hidden_dim))
        self.f_aggr = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), FeaturePReLU(hidden_dim))
        self.f_out = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), FeaturePReLU(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        m = self.f_gen(z)
        a = max_over_others(self.f_aggr(m))
        return self.f_out(torch.cat((m, a), dim=-1))


class SecureBeamGNN(Beamformer):
    KIND = "gnn"

    def __init__(self, num_antennas: int, ref_snr: float, hidden_dim: int, num_layers: int,
                 input_transform: str = "linear") -> None:
        super().__init__(num_antennas, ref_snr, input_transform, hidden_dim=hidden_dim, num_layers=num_layers)
        channel_dim, embed_dim = 4 * num_antennas, 2 * num_antennas
        self.embed_dim = embed_dim
        self.layers = nn.ModuleList(
            GCNLayer(channel_dim + embed_dim, hidden_dim, embed_dim) for _ in range(num_layers)
        )

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        # Channel part of the node feature is preserved across layers; only e_k is updated.
        e = x.new_zeros(*x.shape[:2], self.embed_dim)
        for layer in self.layers:
            e = layer(torch.cat((x, e), dim=-1))
        return e
