"""Unsupervised training of beamforming networks with the loss of eq. (20).

Each mini-batch is a fresh set of network instances: random user/eavesdropper
layouts, a random UAV position inside the area and random small-scale fading.
The loss is the negative mean sum secrecy rate, so no labels are needed.
"""
from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field

import torch

from ..channel import channels, reference_snr, sample_scenario, sample_uav_positions
from ..config import Config, SystemConfig
from ..models import Beamformer, MLPBeamformer, SecureBeamGNN
from ..secrecy import sum_secrecy_rate
from ..utils import make_generator


def build_model(kind: str, cfg: Config, num_users: int | None = None) -> Beamformer:
    """Create a GNN or MLP beamformer from the config (``num_users`` only matters for the MLP)."""
    sys = cfg.system
    if kind == "gnn":
        g = cfg.gnn
        model = SecureBeamGNN(sys.num_antennas, reference_snr(sys), hidden_dim=g.hidden_dim,
                              num_layers=g.num_layers, input_transform=g.input_transform,
                              out_layers=g.out_layers, init_embedding=g.init_embedding,
                              aggregate_self=g.aggregate_self, layer_norm=g.layer_norm, residual=g.residual,
                              input_power=None if g.input_power_scaling else sys.power_max_w)
        model.set_empty_aggregate(g.empty_aggregate)
        return model
    if kind == "mlp":
        return MLPBeamformer(sys.num_antennas, reference_snr(sys), num_users=num_users or sys.num_users,
                             hidden_dims=cfg.mlp.hidden_dims, input_transform=cfg.gnn.input_transform)
    raise ValueError(f"unknown model kind {kind!r}")


def learning_rate(kind: str, cfg: Config) -> float:
    return cfg.gnn.lr if kind == "gnn" else cfg.mlp.lr


UAV_SAMPLING = ("uniform", "center", "near_user")


def make_optimizer(name: str, params, lr: float) -> torch.optim.Optimizer:
    """The paper says only "stochastic gradient descent" (p. 7): "sgd" is SGD with momentum 0.9,
    "adam" is the adaptive variant used by default."""
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    raise ValueError(f"gnn.optimizer must be 'adam' or 'sgd', got {name!r}")


def sample_training_batch(sys: SystemConfig, batch: int, num_users: int, gen: torch.Generator,
                          device: torch.device, sampling: str = "uniform", frac: float = 0.0,
                          std_m: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Random layouts and fading. The UAV is uniform in the area, except for a ``frac`` share of
    the batch drawn from N(anchor, std_m^2) with anchor = area centre ("center", where the UAV
    starts) or a random user ("near_user", where deployment tends to place it)."""
    if sampling not in UAV_SAMPLING:
        raise ValueError(f"uav_sampling must be one of {UAV_SAMPLING}, got {sampling!r}")
    scenario = sample_scenario(sys, batch, gen, device, num_users=num_users)
    uav = sample_uav_positions(sys, batch, gen, device)
    biased = torch.rand(batch, generator=gen, device=device) < (frac if sampling != "uniform" else 0.0)
    if biased.any():
        if sampling == "center":
            anchor = torch.full((batch, 2), sys.area_size / 2, device=device)
        else:
            pick = torch.randint(num_users, (batch,), generator=gen, device=device)
            anchor = scenario.users[torch.arange(batch, device=device), pick]
        around = anchor + std_m * torch.randn(batch, 2, generator=gen, device=device)
        uav = torch.where(biased.unsqueeze(-1), around.clamp(0.0, sys.area_size), uav)
    return channels(sys, uav, scenario, gen)


@torch.no_grad()
def evaluate_rate(model: Beamformer, h: torch.Tensor, h_e: torch.Tensor, power: float,
                  chunk: int = 4096) -> torch.Tensor:
    """Sum secrecy rate per instance (with the (.)^+ clamp), evaluated in chunks."""
    model.eval()
    out = [sum_secrecy_rate(h[i:i + chunk], h_e[i:i + chunk], model(h[i:i + chunk], h_e[i:i + chunk], power))
           for i in range(0, h.shape[0], chunk)]
    return torch.cat(out)


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)   # mean loss per epoch (eq. (20))
    val_rate: list[float] = field(default_factory=list)     # mean sum secrecy rate on the val set
    best_epoch: int = -1
    best_val_rate: float = float("-inf")
    train_time_s: float = 0.0


def train_beamformer(model: Beamformer, cfg: Config, *, num_users: int, epochs: int, lr: float,
                     seed: int, device: torch.device, logger: logging.Logger | None = None,
                     tag: str = "") -> TrainHistory:
    """Train ``model`` in place; on return it holds the weights of the best validation epoch."""
    sys, g = cfg.system, cfg.gnn
    model.to(device)
    optimizer = make_optimizer(g.optimizer, model.parameters(), lr)
    # The paper gives the learning rate (0.005) but no schedule; "cosine" starts there and decays to 1%.
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=0.01 * lr)
                 if g.lr_schedule == "cosine" else None)
    train_gen = make_generator(seed, device)
    val_h, val_he = sample_training_batch(sys, g.val_samples, num_users,
                                          make_generator(seed + 10_000, device), device)
    history = TrainHistory()
    best_state = copy.deepcopy(model.state_dict())
    start = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        losses = []
        for _ in range(g.batches_per_epoch):
            h, h_e = sample_training_batch(sys, g.batch_size, num_users, train_gen, device,
                                           g.uav_sampling, g.uav_sampling_frac, g.uav_sampling_std_m)
            w = model(h, h_e, sys.power_max_w)
            loss = -sum_secrecy_rate(h, h_e, w, clamp=g.clamp_loss).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), g.grad_clip)
            optimizer.step()
            losses.append(loss.item())

        if scheduler is not None:
            scheduler.step()
        val = evaluate_rate(model, val_h, val_he, sys.power_max_w).mean().item()
        history.train_loss.append(sum(losses) / len(losses))
        history.val_rate.append(val)
        if val > history.best_val_rate:
            history.best_val_rate, history.best_epoch = val, epoch
            best_state = copy.deepcopy(model.state_dict())
        if logger and (epoch % 10 == 0 or epoch == epochs - 1):
            logger.info("%s epoch %4d | loss %8.3f | val sum-secrecy %7.3f bps/Hz",
                        tag, epoch, history.train_loss[-1], val)

    history.train_time_s = time.perf_counter() - start
    model.load_state_dict(best_state)
    model.eval()
    return history
