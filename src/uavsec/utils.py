"""Small shared helpers: seeding, devices, run directories, logging, I/O."""
from __future__ import annotations

import json
import logging
import random
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import yaml

from .config import Config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_generator(seed: int, device: torch.device) -> torch.Generator:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen


def git_hash() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, check=True, cwd=Path(__file__).parent)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(obj), indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass
class Run:
    """An experiment run directory with config snapshot, log file and metrics.

    Layout: ``<output_dir>/<name>_<timestamp>/{config.yaml, log.txt, metrics.json, ...}``
    """
    name: str
    cfg: Config
    dir: Path = field(init=False)
    logger: logging.Logger = field(init=False)
    _start: float = field(init=False)

    def __post_init__(self) -> None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = self.cfg.output_path / f"{self.name}_{stamp}"
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "config.yaml").write_text(yaml.safe_dump(self.cfg.to_dict(), sort_keys=False),
                                              encoding="utf-8")
        self.logger = _make_logger(self.name, self.dir / "log.txt")
        self._start = time.perf_counter()
        self.logger.info("run dir: %s | git %s", self.dir, git_hash())

    def path(self, *parts: str) -> Path:
        p = self.dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def finish(self, metrics: dict[str, Any]) -> Path:
        """Write metrics.json with provenance info and return its path."""
        payload = {
            "experiment": self.name,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "git": git_hash(),
            "seed": self.cfg.seed,
            "wall_time_s": time.perf_counter() - self._start,
            **metrics,
        }
        out = self.path("metrics.json")
        save_json(payload, out)
        self.logger.info("metrics written to %s", out)
        return out


def _make_logger(name: str, file: Path) -> logging.Logger:
    logger = logging.getLogger(f"uavsec.{name}.{file.parent.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(file, encoding="utf-8")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def latest_run(output_dir: str | Path, name: str) -> Path | None:
    """Most recent run directory of an experiment that finished (has metrics.json)."""
    # Match "<name>_<YYYYmmdd-HHMMSS>" exactly, so "deployment" does not pick up "deployment_sensitivity_*".
    pattern = re.compile(rf"{re.escape(name)}_\d{{8}}-\d{{6}}")
    runs = sorted(p for p in Path(output_dir).glob(f"{name}_*")
                  if pattern.fullmatch(p.name) and (p / "metrics.json").exists())
    return runs[-1] if runs else None


def find_checkpoint(cfg: Config, filename: str, explicit: str | None = None) -> Path | None:
    """An explicit checkpoint path, else ``ckpt/<filename>`` of the latest convergence run."""
    if explicit:
        return Path(explicit)
    run_dir = latest_run(cfg.output_path, "convergence")
    candidate = run_dir / "ckpt" / filename if run_dir else None
    return candidate if candidate is not None and candidate.exists() else None


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    """Wall-clock timer that synchronises CUDA so GPU work is fully counted."""
    out: dict[str, float] = {}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    yield out
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    out["seconds"] = time.perf_counter() - start
