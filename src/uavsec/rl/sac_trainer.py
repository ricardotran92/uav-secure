"""SAC training (Section V-B/C, Algorithm 1) on top of Stable-Baselines3.

SB3's SAC implements eqs. (29)-(36): twin critics with target networks, soft
updates, a squashed-Gaussian actor and automatic entropy-temperature tuning.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from ..config import SACConfig
from ..envs.uav_env import UAVDeploymentEnv


class EpisodeLogger(BaseCallback):
    """Collects per-episode reward, mean secrecy rate and final UAV position."""

    def __init__(self, logger: logging.Logger | None = None, log_every: int = 10) -> None:
        super().__init__()
        self.records: list[dict] = []
        self._log, self._every = logger, log_every
        self._reward = 0.0
        self._rates: list[float] = []

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        self._reward += float(self.locals["rewards"][0])
        self._rates.append(info["secrecy_rate"])
        if self.locals["dones"][0]:
            x, y = info["position"]
            self.records.append({"episode": len(self.records), "episode_reward": self._reward,
                                 "mean_secrecy_rate": float(np.mean(self._rates)),
                                 "final_x": x, "final_y": y, "timesteps": self.num_timesteps})
            if self._log and len(self.records) % self._every == 0:
                r = self.records[-1]
                self._log.info("episode %4d | reward %9.2f | mean secrecy %7.3f | end (%.1f, %.1f)",
                               r["episode"], r["episode_reward"], r["mean_secrecy_rate"], x, y)
            self._reward, self._rates = 0.0, []
        return True

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)


def train_sac(env: UAVDeploymentEnv, cfg: SACConfig, device: torch.device, seed: int,
              logger: logging.Logger | None = None) -> tuple[SAC, pd.DataFrame]:
    model = SAC(
        "MlpPolicy", env, learning_rate=cfg.lr, buffer_size=cfg.buffer_size,
        learning_starts=cfg.learning_starts, batch_size=cfg.batch_size, tau=cfg.tau, gamma=cfg.gamma,
        policy_kwargs={"net_arch": list(cfg.net_arch)}, seed=seed, device=device, verbose=0,
    )
    callback = EpisodeLogger(logger)
    model.learn(total_timesteps=cfg.episodes * cfg.episode_length, callback=callback)
    return model, callback.frame()


def rollout(model: SAC, env: UAVDeploymentEnv, seed: int, deterministic: bool = True) -> pd.DataFrame:
    """Run one episode from the area centre; returns the trajectory with per-step rates."""
    obs, info = env.reset(seed=seed)
    rows = [{"step": 0, "x": info["position"][0], "y": info["position"][1], "secrecy_rate": float(env.secrecy().sum())}]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        rows.append({"step": env.t, "x": info["position"][0], "y": info["position"][1], "secrecy_rate": reward})
        done = terminated or truncated
    return pd.DataFrame(rows)
