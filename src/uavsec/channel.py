"""Network geometry and air-to-ground channel model (Section III, Section VI setup).

Conventions
-----------
* All tensors are batched: ``B`` instances, ``K`` user/eavesdropper pairs,
  ``N`` UAV antennas. Positions are horizontal ``(x, y)`` in metres.
* Channels are returned **noise-normalised**, i.e. ``h / sigma_0``, so that the
  receiver noise power is 1 in every rate expression. This keeps float32
  numerics well conditioned (raw gains are ~1e-8 against noise ~1e-13) and
  makes noise-power sweeps a simple rescaling of the channel.
* Channel model: Rician fading with factor ``kappa``; LoS component is the
  steering vector of a half-wavelength ULA at the UAV, path loss
  ``PL(d) = a + b log10(d)`` dB with ``d`` the 3-D UAV-node distance.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from .config import REPO_ROOT, SystemConfig


AXES = {"x": 0, "y": 1}          # ULA orientation -> coordinate used for the direction cosine
DISTANCE_MODELS = ("3d", "horizontal")


@dataclass
class Scenario:
    """Ground-node layout of a batch of network instances."""
    users: torch.Tensor  # (B, K, 2)
    eves: torch.Tensor   # (B, K, 2)

    @property
    def batch_size(self) -> int:
        return self.users.shape[0]

    def expand(self, n: int) -> "Scenario":
        """Repeat a single-instance scenario ``n`` times (for position sweeps)."""
        assert self.batch_size == 1, "expand() expects a single instance"
        return Scenario(self.users.expand(n, -1, -1), self.eves.expand(n, -1, -1))


def sample_scenario(sys: SystemConfig, batch: int, gen: torch.Generator,
                    device: torch.device, num_users: int | None = None) -> Scenario:
    """Users uniform in the area; each eavesdropper ~eve_distance away at a random bearing."""
    k = num_users or sys.num_users
    users = torch.rand(batch, k, 2, generator=gen, device=device) * sys.area_size
    bearing = torch.rand(batch, k, generator=gen, device=device) * 2 * math.pi
    offset = sys.eve_distance * torch.stack((bearing.cos(), bearing.sin()), dim=-1)
    eves = (users + offset).clamp(0.0, sys.area_size)
    return Scenario(users, eves)


def _layout_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_paper_reference(path: str | Path) -> dict | None:
    """Optional ``paper_reference`` block of a layout file (digitised results to compare against)."""
    with open(_layout_path(path), encoding="utf-8") as fh:
        return yaml.safe_load(fh).get("paper_reference")


def load_layout(path: str | Path, sys: SystemConfig, device: torch.device) -> Scenario:
    """Fixed single-instance layout from a YAML file with ``users`` and ``eves`` lists of [x, y].

    ``eves[k]`` is the eavesdropper of ``users[k]``. Counts and bounds are validated
    against the system config so a wrong file fails loudly.
    """
    path = _layout_path(path)
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    users = torch.tensor(data["users"], dtype=torch.float32, device=device)
    eves = torch.tensor(data["eves"], dtype=torch.float32, device=device)
    if users.shape != (sys.num_users, 2) or eves.shape != users.shape:
        raise ValueError(f"{path}: expected {sys.num_users} users and as many eavesdroppers, "
                         f"got {tuple(users.shape)} and {tuple(eves.shape)}")
    if not (((users >= 0) & (users <= sys.area_size)).all() and ((eves >= 0) & (eves <= sys.area_size)).all()):
        raise ValueError(f"{path}: all nodes must lie inside the {sys.area_size:g} m area")
    return Scenario(users.unsqueeze(0), eves.unsqueeze(0))


def sample_uav_positions(sys: SystemConfig, batch: int, gen: torch.Generator,
                         device: torch.device) -> torch.Tensor:
    return torch.rand(batch, 2, generator=gen, device=device) * sys.area_size


def area_center(sys: SystemConfig, batch: int, device: torch.device) -> torch.Tensor:
    return torch.full((batch, 2), sys.area_size / 2, device=device)


def path_loss_db(sys: SystemConfig, distance: torch.Tensor | float) -> torch.Tensor:
    """PL(d) = a + b log(d) dB, with the logarithm base set by ``path_loss_log``."""
    distance = torch.as_tensor(distance, dtype=torch.float32)
    log = torch.log10 if sys.path_loss_log == "log10" else torch.log
    return sys.path_loss_a_db + sys.path_loss_b * log(distance)


def path_gain(sys: SystemConfig, distance: torch.Tensor) -> torch.Tensor:
    """Linear large-scale power gain 10^(-PL(d)/10)."""
    return 10.0 ** (-path_loss_db(sys, distance) / 10.0)


def steering_vector(direction_cos: torch.Tensor, num_antennas: int, spacing: float) -> torch.Tensor:
    """ULA response a_n = exp(j 2 pi s n cos(phi)), n = 0..N-1.  Shape (..., N)."""
    n = torch.arange(num_antennas, device=direction_cos.device, dtype=direction_cos.dtype)
    phase = 2 * math.pi * spacing * direction_cos.unsqueeze(-1) * n
    return torch.polar(torch.ones_like(phase), phase)


def complex_normal(shape: tuple[int, ...], gen: torch.Generator, device: torch.device) -> torch.Tensor:
    """i.i.d. CN(0, 1) samples."""
    re = torch.randn(shape, generator=gen, device=device)
    im = torch.randn(shape, generator=gen, device=device)
    return torch.complex(re, im) / math.sqrt(2.0)


def rician_channels(sys: SystemConfig, uav: torch.Tensor, nodes: torch.Tensor,
                    gen: torch.Generator, noise_power_w: float | None = None) -> torch.Tensor:
    """Noise-normalised channels from the UAV at ``uav`` (B, 2) to ``nodes`` (B, K, 2).

    Returns complex (B, K, N): sqrt(beta/sigma^2) * (sqrt(k/(k+1)) a + sqrt(1/(k+1)) g).
    """
    sigma2 = sys.noise_power_w if noise_power_w is None else noise_power_w
    delta = nodes - uav.unsqueeze(1)                                   # (B, K, 2)
    horizontal = delta.norm(dim=-1)                                    # (B, K)
    dist_3d = torch.sqrt(horizontal ** 2 + sys.uav_altitude ** 2)
    # The LoS direction is always geometric (3-D); only the path-loss distance is configurable.
    pl_dist = dist_3d if sys.distance_model == "3d" else horizontal.clamp_min(1.0)
    axis = AXES[sys.array_axis]
    los = steering_vector(delta[..., axis] / dist_3d, sys.num_antennas, sys.antenna_spacing)
    nlos = complex_normal(los.shape, gen, uav.device)
    kappa = 10.0 ** (sys.rician_factor_db / 10.0)
    small_scale = math.sqrt(kappa / (kappa + 1)) * los + math.sqrt(1 / (kappa + 1)) * nlos
    amplitude = torch.sqrt(path_gain(sys, pl_dist) / sigma2).unsqueeze(-1)
    return amplitude * small_scale


def channels(sys: SystemConfig, uav: torch.Tensor, scenario: Scenario, gen: torch.Generator,
             noise_power_w: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return noise-normalised (h, h_E), each complex (B, K, N)."""
    h = rician_channels(sys, uav, scenario.users, gen, noise_power_w)
    h_e = rician_channels(sys, uav, scenario.eves, gen, noise_power_w)
    return h, h_e


def reference_snr(sys: SystemConfig) -> float:
    """P^max * beta(H) / sigma^2 for a node directly below the UAV (default setting).

    Used as a fixed input scale for the neural networks so that their inputs are
    O(1) at the training setting while still reflecting changes of P or sigma^2.
    """
    beta = float(path_gain(sys, torch.tensor(sys.uav_altitude)))
    return sys.power_max_w * beta / sys.noise_power_w
