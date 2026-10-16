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


EMPTY_AGGREGATE = ("zero", "self")


def max_over_others(x: torch.Tensor, empty: str = "zero") -> torch.Tensor:
    """a_k = max_{l != k} x_l along the node axis. x: (B, K, H) -> (B, K, H).

    For K = 1 the neighbour set of eq. (16) is empty, which the paper does not define:
    ``empty="zero"`` returns zeros, ``empty="self"`` returns the node's own message.
    """
    k = x.shape[1]
    if k == 1:
        return x if empty == "self" else torch.zeros_like(x)
    # Top-2 trick: the max over the others is the overall max, except for the node that
    # holds it, which gets the second largest. O(K) memory instead of O(K^2).
    top, idx = x.topk(2, dim=1)                                 # (B, 2, H)
    node = torch.arange(k, device=x.device).reshape(1, k, 1)
    return torch.where(node == idx[:, :1], top[:, 1:2], top[:, :1])


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, out_layers: int = 2) -> None:
        super().__init__()
        self.f_gen = nn.Sequential(nn.Linear(in_dim, hidden_dim), FeaturePReLU(hidden_dim))
        self.f_aggr = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), FeaturePReLU(hidden_dim))
        # f_out is "a multilayer perceptron" (eq. (18)); its depth is not given in the paper.
        blocks: list[nn.Module] = [nn.Linear(2 * hidden_dim, hidden_dim), FeaturePReLU(hidden_dim)]
        for _ in range(out_layers - 2):
            blocks += [nn.Linear(hidden_dim, hidden_dim), FeaturePReLU(hidden_dim)]
        self.f_out = nn.Sequential(*blocks, nn.Linear(hidden_dim, out_dim))

        self.empty_aggregate = "zero"                       # rule for K = 1 (no neighbours)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        m = self.f_gen(z)
        a = max_over_others(self.f_aggr(m), self.empty_aggregate)
        return self.f_out(torch.cat((m, a), dim=-1))


class SecureBeamGNN(Beamformer):
    KIND = "gnn"

    def __init__(self, num_antennas: int, ref_snr: float, hidden_dim: int, num_layers: int,
                 input_transform: str = "linear", out_layers: int = 2, init_embedding: str = "zeros") -> None:
        if init_embedding not in ("zeros", "mrt"):
            raise ValueError(f"init_embedding must be 'zeros' or 'mrt', got {init_embedding!r}")
        super().__init__(num_antennas, ref_snr, input_transform, hidden_dim=hidden_dim, num_layers=num_layers,
                         out_layers=out_layers, init_embedding=init_embedding)
        channel_dim, embed_dim = 4 * num_antennas, 2 * num_antennas
        self.embed_dim = embed_dim
        self.init_embedding = init_embedding
        self.layers = nn.ModuleList(
            GCNLayer(channel_dim + embed_dim, hidden_dim, embed_dim, out_layers) for _ in range(num_layers)
        )

    def set_empty_aggregate(self, mode: str) -> None:
        """Inference rule for K = 1 (no learned parameters involved; K >= 2 is unaffected)."""
        if mode not in EMPTY_AGGREGATE:
            raise ValueError(f"empty_aggregate must be one of {EMPTY_AGGREGATE}, got {mode!r}")
        for layer in self.layers:
            layer.empty_aggregate = mode

    def initial_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """e_k^(0): zeros, or the MRT direction h_k / ||h_k|| (real and imaginary parts)."""
        if self.init_embedding == "zeros":
            return x.new_zeros(*x.shape[:2], self.embed_dim)
        h = x[..., :self.embed_dim]                          # [Re h_k, Im h_k] of eq. (12)
        return h / h.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        # Channel part of the node feature is preserved across layers; only e_k is updated.
        e = self.initial_embedding(x)
        for layer in self.layers:
            e = layer(torch.cat((x, e), dim=-1))
        return e
