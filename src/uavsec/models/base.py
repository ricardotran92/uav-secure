"""Common interface for learned beamformers: channels in, power-normalised W out."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..secrecy import normalize_power

INPUT_TRANSFORMS = ("linear", "log")


class Beamformer(nn.Module):
    """Base class. Subclasses implement ``embed`` returning real (B, K, 2N) embeddings.

    ``ref_snr`` (see ``channel.reference_snr``) is a fixed input scale: features are
    sqrt(P / ref_snr) * h, which is O(1) at the training setting and still tracks
    changes in transmit or noise power at test time. With ``input_power`` set, the
    features use that fixed power instead of P (the input no longer depends on the
    test power, which then enters only through the power normalisation of W).
    """

    def __init__(self, num_antennas: int, ref_snr: float, input_transform: str = "linear",
                 input_power: float | None = None, **hparams: Any) -> None:
        super().__init__()
        if input_transform not in INPUT_TRANSFORMS:
            raise ValueError(f"input_transform must be one of {INPUT_TRANSFORMS}, got {input_transform!r}")
        self.num_antennas = num_antennas
        self.ref_snr = ref_snr
        self.input_transform = input_transform
        self.input_power = input_power
        self.hparams = {"num_antennas": num_antennas, "ref_snr": ref_snr,
                        "input_transform": input_transform, **hparams}
        if input_power is not None:
            self.hparams["input_power"] = input_power

    def _scale(self, h: torch.Tensor, power: float | torch.Tensor) -> torch.Tensor:
        """sqrt(P / SNR_ref) * h, optionally with log-compressed magnitude (phase unchanged).

        "log" maps |h| -> log(1 + |h|): needed when the channel gain spans many decades,
        e.g. with a horizontal-distance path loss where the UAV can hover right above a node.
        """
        if self.input_power is not None:
            power = self.input_power
        scale = torch.sqrt(torch.as_tensor(power, dtype=torch.float32, device=h.device) / self.ref_snr)
        h = h * scale.reshape(-1, 1, 1)
        if self.input_transform == "log":
            mag = h.abs().clamp_min(1e-12)
            h = h * (torch.log1p(mag) / mag)
        return h

    def features(self, h: torch.Tensor, h_e: torch.Tensor, power: float | torch.Tensor) -> torch.Tensor:
        """Node features [Re h_k, Im h_k, Re h_E,k, Im h_E,k] of eq. (12), shape (B, K, 4N)."""
        h, h_e = self._scale(h, power), self._scale(h_e, power)
        return torch.cat((h.real, h.imag, h_e.real, h_e.imag), dim=-1)

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
