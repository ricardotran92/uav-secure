"""Common interface for learned beamformers: channels in, power-normalised W out."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..secrecy import normalize_power


class Beamformer(nn.Module):
    """Base class. Subclasses implement ``embed`` returning real (B, K, 2N) embeddings.

    ``ref_snr`` (see ``channel.reference_snr``) is a fixed input scale: features are
    sqrt(P / ref_snr) * h, which is O(1) at the training setting and still tracks
    changes in transmit or noise power at test time.
    """

    def __init__(self, num_antennas: int, ref_snr: float, **hparams: Any) -> None:
        super().__init__()
        self.num_antennas = num_antennas
        self.ref_snr = ref_snr
        self.hparams = {"num_antennas": num_antennas, "ref_snr": ref_snr, **hparams}

    def features(self, h: torch.Tensor, h_e: torch.Tensor, power: float | torch.Tensor) -> torch.Tensor:
        """Node features [Re h_k, Im h_k, Re h_E,k, Im h_E,k] of eq. (12), shape (B, K, 4N)."""
        scale = torch.sqrt(torch.as_tensor(power, dtype=torch.float32, device=h.device) / self.ref_snr)
        x = torch.cat((h.real, h.imag, h_e.real, h_e.imag), dim=-1)
        return x * scale.reshape(-1, 1, 1)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, h: torch.Tensor, h_e: torch.Tensor, power: float | torch.Tensor) -> torch.Tensor:
        e = self.embed(self.features(h, h_e, power))
        n = self.num_antennas
        return normalize_power(torch.complex(e[..., :n], e[..., n:]), power)


class FeaturePReLU(nn.Module):
    """PReLU with one learnable slope per feature on the *last* axis.

    ``nn.PReLU(num_parameters)`` expects channels on axis 1, which does not fit
    node tensors shaped (B, K, H).
    """

    def __init__(self, num_features: int, init: float = 0.25) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.full((num_features,), init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x >= 0, x, self.weight * x)


def save_model(model: Beamformer, path: Path, **extra: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": model.KIND, "hparams": model.hparams, "state_dict": model.state_dict(), **extra}, path)


def load_model(path: Path, device: torch.device | str = "cpu") -> Beamformer:
    from .gnn import SecureBeamGNN
    from .mlp import MLPBeamformer

    registry = {cls.KIND: cls for cls in (SecureBeamGNN, MLPBeamformer)}
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = registry[ckpt["kind"]](**ckpt["hparams"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def dbm_to_watt(dbm: float) -> float:
    return 10.0 ** (dbm / 10.0) / 1000.0


def watt_to_dbm(w: float) -> float:
    return 10.0 * math.log10(w * 1000.0)
