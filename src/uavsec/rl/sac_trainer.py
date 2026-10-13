"""SAC training (Section V-B/C, Algorithm 1) on top of Stable-Baselines3.

SB3's SAC implements eqs. (29)-(36): twin critics with target networks, soft
updates, a squashed-Gaussian actor and automatic entropy-temperature tuning.
"""
from __future__ import annotations

import copy
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
        self._reward += info["secrecy_rate"]          # unscaled R_sec, as in the paper's Fig. 6
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


class BestPolicyCallback(BaseCallback):
    """Every ``every`` episodes, run one deterministic episode on a separate environment and
    keep the policy whose UAV settles at the highest secrecy rate (mean of the last 20% of steps).
    """

    def __init__(self, eval_env: UAVDeploymentEnv, every: int, seed: int,
                 logger: logging.Logger | None = None) -> None:
        super().__init__()
        self.eval_env, self.every, self.seed, self._log = eval_env, every, seed, logger
        self.records: list[dict] = []
        self.best_score = float("-inf")
        self._best_state: dict | None = None
        self._episodes = 0

    def _on_step(self) -> bool:
        if self.locals["dones"][0]:
            self._episodes += 1
            if self._episodes % self.every == 0:
                self._evaluate()
        return True

    def _evaluate(self) -> None:
        traj = rollout(self.model, self.eval_env, seed=self.seed)
        score = float(traj["secrecy_rate"].tail(max(1, len(traj) // 5)).mean())
        x, y = traj[["x", "y"]].to_numpy()[-1]
        self.records.append({"episode": self._episodes, "eval_score": score, "end_x": x, "end_y": y})
        if score > self.best_score:
            self.best_score = score
            self._best_state = copy.deepcopy(self.model.policy.state_dict())
            if self._log:
                self._log.info("new best policy at episode %d: settles at (%.1f, %.1f), %.2f bps/Hz",
                               self._episodes, x, y, score)

    def restore_best(self) -> None:
        if self._best_state is not None:
            self.model.policy.load_state_dict(self._best_state)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)


def train_sac(env: UAVDeploymentEnv, eval_env: UAVDeploymentEnv, cfg: SACConfig, device: torch.device,
              seed: int, logger: logging.Logger | None = None) -> tuple[SAC, pd.DataFrame, pd.DataFrame]:
    """Train SAC; returns the best evaluated policy, per-episode training log and evaluation log."""
    model = SAC(
        "MlpPolicy", env, learning_rate=cfg.lr, buffer_size=cfg.buffer_size,
        learning_starts=cfg.learning_starts, batch_size=cfg.batch_size, tau=cfg.tau, gamma=cfg.gamma,
        policy_kwargs={"net_arch": list(cfg.net_arch)}, seed=seed, device=device, verbose=0,
    )
    episodes = EpisodeLogger(logger)
    best = BestPolicyCallback(eval_env, cfg.eval_every_episodes, seed + 1, logger)
    model.learn(total_timesteps=cfg.episodes * cfg.episode_length, callback=[episodes, best])
    best.restore_best()
    return model, episodes.frame(), best.frame()


def rollout(model: SAC, env: UAVDeploymentEnv, seed: int, deterministic: bool = True) -> pd.DataFrame:
    """Run one episode from the area centre; returns the trajectory with per-step rates."""
    obs, info = env.reset(seed=seed)
    rows = [{"step": 0, "x": info["position"][0], "y": info["position"][1], "secrecy_rate": float(env.secrecy().sum())}]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, info = env.step(action)
        rows.append({"step": env.t, "x": info["position"][0], "y": info["position"][1],
                     "secrecy_rate": info["secrecy_rate"]})
        done = terminated or truncated
    return pd.DataFrame(rows)
