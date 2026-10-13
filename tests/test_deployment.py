"""Deployment heuristics and the Gymnasium environment."""
import numpy as np
import pytest
import torch

from uavsec.baselines.deployment import heuristic_positions, min_enclosing_circle_center, polygon_centroid
from uavsec.channel import reference_snr, sample_scenario
from uavsec.config import DEFAULT_CONFIG, load_config
from uavsec.envs.uav_env import UAVDeploymentEnv
from uavsec.models import SecureBeamGNN
from uavsec.utils import make_generator

DEVICE = torch.device("cpu")


def test_geometric_heuristics_on_known_shapes():
    square = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.5, 0.5]])
    assert np.allclose(min_enclosing_circle_center(square), [1.0, 1.0])
    assert np.allclose(polygon_centroid(square), [1.0, 1.0])              # interior point ignored
    triangle = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]])
    assert np.allclose(min_enclosing_circle_center(triangle), [2.0, 1.5])  # right triangle: hypotenuse midpoint
    collinear = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    assert np.allclose(polygon_centroid(collinear), [1.0, 1.0])


def test_heuristics_inside_area():
    users = np.random.default_rng(0).uniform(0, 200, size=(8, 2))
    for name, pos in heuristic_positions(users, 200.0).items():
        assert np.all((pos >= 0) & (pos <= 200)), name


@pytest.fixture
def env():
    cfg = load_config([DEFAULT_CONFIG], ["sac.episode_length=5", "sac.max_step_m=50"])
    gnn = SecureBeamGNN(cfg.system.num_antennas, reference_snr(cfg.system), hidden_dim=16, num_layers=1)
    scenario = sample_scenario(cfg.system, 1, make_generator(0, DEVICE), DEVICE)
    return UAVDeploymentEnv(cfg, gnn, scenario, DEVICE)


def test_env_episode(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert np.allclose(info["position"], [100.0, 100.0])
    for t in range(5):
        obs, reward, terminated, truncated, info = env.step(np.array([1.0, 1.0], dtype=np.float32))
        assert reward >= 0 and not terminated
        assert reward == pytest.approx(env.sac.reward_scale * info["secrecy_rate"])
        assert truncated == (t == 4)
    assert np.allclose(info["position"], [200.0, 200.0])                   # clipped to the area


def test_env_seeded_reset_is_reproducible(env):
    obs1, _ = env.reset(seed=3)
    obs2, _ = env.reset(seed=3)
    assert np.allclose(obs1, obs2)
