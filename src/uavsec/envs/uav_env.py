"""MDP of Section V-A as a Gymnasium environment.

State  s(t) = {(x_U, y_U), {h_k}, {h_E,k}}            (26)
Action a(t) = (dx, dy) in [-1, 1]^2, scaled to metres  (27)-(28)
Reward r(t) = sum secrecy rate with GNN beamforming at the new position.

Ground nodes are fixed within a run (as in Fig. 7); small-scale fading is
redrawn at every step. The position is clipped to the area (constraint (25)).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from ..channel import Scenario, channels
from ..config import Config
from ..models import Beamformer
from ..secrecy import secrecy_rates
from ..utils import make_generator


class UAVDeploymentEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: Config, beamformer: Beamformer, scenario: Scenario, device: torch.device,
                 seed: int = 0) -> None:
        super().__init__()
        self.sys, self.sac = cfg.system, cfg.sac
        self.beamformer = beamformer.eval()
        self.scenario = scenario                     # single instance (1, K, 2)
        self.device = device
        self.gen = make_generator(seed, device)
        k, n = scenario.users.shape[1], self.sys.num_antennas
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(2 + 4 * n * k,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.position = np.zeros(2)
        self.t = 0
        self._h = self._h_e = None

    def _sample_channels(self) -> None:
        uav = torch.as_tensor(self.position, dtype=torch.float32, device=self.device).unsqueeze(0)
        self._h, self._h_e = channels(self.sys, uav, self.scenario, self.gen)

    def _observation(self) -> np.ndarray:
        # Position mapped to [-1, 1]; channels scaled exactly as the GNN sees them.
        pos = self.position / self.sys.area_size * 2 - 1
        feats = self.beamformer.features(self._h, self._h_e, self.sys.power_max_w)
        return np.concatenate([pos, feats.flatten().cpu().numpy()]).astype(np.float32)

    @torch.no_grad()
    def secrecy(self) -> np.ndarray:
        """Per-user secrecy rates at the current position and fading state."""
        w = self.beamformer(self._h, self._h_e, self.sys.power_max_w)
        return secrecy_rates(self._h, self._h_e, w)[0].cpu().numpy()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.gen.manual_seed(seed)
        start = (options or {}).get("start")
        self.position = np.asarray(start, dtype=float) if start is not None else np.full(2, self.sys.area_size / 2)
        self.t = 0
        self._sample_channels()
        return self._observation(), {"position": self.position.copy()}

    def step(self, action: np.ndarray):
        delta = np.clip(action, -1.0, 1.0) * self.sac.max_step_m
        self.position = np.clip(self.position + delta, 0.0, self.sys.area_size)
        self.t += 1
        self._sample_channels()
        per_user = self.secrecy()
        reward = float(per_user.sum())
        truncated = self.t >= self.sac.episode_length
        info = {"position": self.position.copy(), "secrecy_rate": reward, "per_user_secrecy": per_user}
        return self._observation(), reward, False, truncated, info
