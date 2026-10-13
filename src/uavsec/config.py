"""Typed experiment configuration.

The YAML files in ``configs/`` are the single source of truth: the dataclasses
below carry no defaults, so a missing or misspelled key fails loudly instead of
silently falling back to a hidden value.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class SystemConfig:
    area_size: float
    num_users: int
    num_antennas: int
    uav_altitude: float
    rician_factor_db: float
    bandwidth_hz: float
    power_max_w: float
    noise_power_w: float
    path_loss_a_db: float
    path_loss_b: float
    path_loss_log: str
    eve_distance: float
    antenna_spacing: float
    distance_model: str
    array_axis: str


@dataclass
class GNNConfig:
    num_layers: int
    hidden_dim: int
    lr: float
    epochs: int
    batch_size: int
    batches_per_epoch: int
    val_samples: int
    clamp_loss: bool
    grad_clip: float
    lr_schedule: str
    input_transform: str


@dataclass
class MLPConfig:
    hidden_dims: list[int]
    lr: float


@dataclass
class SACConfig:
    lr: float
    episodes: int
    episode_length: int
    batch_size: int
    buffer_size: int
    gamma: float
    tau: float
    learning_starts: int
    net_arch: list[int]
    max_step_m: float
    topology_seed: int
    layout_file: str
    obs_transform: str
    reward_scale: float
    eval_every_episodes: int
    gradient_steps: int
    eval_episodes: int


@dataclass
class OptConfig:
    restarts: int
    iters: int
    lr: float


@dataclass
class EvalConfig:
    test_samples: int
    fading_samples: int
    grid_step_m: float
    search_refine_top: int
    search_beamformer: str
    search_opt_restarts: int
    search_opt_iters: int
    uav_position: str
    sweep_users: list[int]
    sweep_power_dbm: list[float]
    sweep_noise_w: list[float]
    cdf_users: list[int]
    timing_users: list[int]
    timing_runs: int
    transfer_users: int
    transfer_epochs: int
    convergence_seeds: list[int]


@dataclass
class Config:
    seed: int
    device: str
    output_dir: str
    system: SystemConfig
    gnn: GNNConfig
    mlp: MLPConfig
    sac: SACConfig
    opt: OptConfig
    eval: EvalConfig

    def __post_init__(self) -> None:
        if self.system.distance_model not in ("3d", "horizontal"):
            raise ValueError(f"system.distance_model must be '3d' or 'horizontal', got {self.system.distance_model!r}")
        if self.system.path_loss_log not in ("log10", "ln"):
            raise ValueError(f"system.path_loss_log must be 'log10' or 'ln', got {self.system.path_loss_log!r}")
        if self.gnn.lr_schedule not in ("constant", "cosine"):
            raise ValueError(f"gnn.lr_schedule must be 'constant' or 'cosine', got {self.gnn.lr_schedule!r}")
        if self.system.array_axis not in ("x", "y"):
            raise ValueError(f"system.array_axis must be 'x' or 'y', got {self.system.array_axis!r}")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def replace(self, **sections: dict[str, Any]) -> "Config":
        """Return a copy with some fields of the given sections changed.

        Example: ``cfg.replace(system={"num_users": 12})``.
        """
        data = self.to_dict()
        for section, values in sections.items():
            if isinstance(values, dict):
                data[section].update(values)
            else:
                data[section] = values
        return _build(Config, data)

    @property
    def output_path(self) -> Path:
        """``output_dir`` resolved against the repository root (if relative)."""
        path = Path(self.output_dir)
        return path if path.is_absolute() else REPO_ROOT / path


def _coerce(value: Any, typ: Any) -> Any:
    """Convert YAML scalars to the annotated type (e.g. '1.2e-13' -> float)."""
    if typ is float:
        return float(value)
    if typ is int:
        return int(value)
    if typ is bool:
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes"}
        return bool(value)
    origin = getattr(typ, "__origin__", None)
    if origin is list:
        (item_type,) = typ.__args__
        return [_coerce(v, item_type) for v in value]
    return value


def _build(cls: type, data: dict[str, Any]) -> Any:
    hints = get_type_hints(cls)
    names = {f.name for f in fields(cls)}
    unknown = set(data) - names
    missing = names - set(data)
    if unknown:
        raise KeyError(f"{cls.__name__}: unknown keys {sorted(unknown)}")
    if missing:
        raise KeyError(f"{cls.__name__}: missing keys {sorted(missing)}")
    kwargs = {}
    for name in names:
        typ = hints[name]
        if is_dataclass(typ):
            kwargs[name] = _build(typ, data[name])
        else:
            kwargs[name] = _coerce(data[name], typ)
    return cls(**kwargs)


def _deep_merge(base: dict, update: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _apply_override(data: dict, override: str) -> None:
    """Apply a ``section.key=value`` override; the value is parsed as YAML."""
    path, _, raw = override.partition("=")
    if not raw:
        raise ValueError(f"override must look like key=value, got {override!r}")
    node = data
    *parents, leaf = path.split(".")
    for key in parents:
        node = node[key]
    if leaf not in node:
        raise KeyError(f"unknown config key {path!r}")
    node[leaf] = yaml.safe_load(raw)


def load_config(paths: list[str | Path], overrides: list[str] | None = None) -> Config:
    """Merge YAML files left-to-right, then apply ``key=value`` overrides."""
    data: dict[str, Any] = {}
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            data = _deep_merge(data, yaml.safe_load(fh) or {})
    for override in overrides or []:
        _apply_override(data, override)
    return _build(Config, data)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", nargs="+", default=[str(DEFAULT_CONFIG)],
                        help="YAML files merged left to right (default: configs/default.yaml)")
    parser.add_argument("--set", nargs="*", default=[], action="extend", metavar="KEY=VALUE",
                        help="overrides, e.g. --set system.num_users=12 gnn.epochs=50")
    parser.add_argument("--quick", action="store_true",
                        help="merge configs/quick.yaml for a fast smoke test")


def config_from_args(args: argparse.Namespace) -> Config:
    paths = list(args.config)
    if args.quick:
        paths.append(str(DEFAULT_CONFIG.parent / "quick.yaml"))
    return load_config(paths, args.set)
