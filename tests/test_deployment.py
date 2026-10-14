"""Deployment heuristics and the Gymnasium environment."""
import numpy as np
import pytest
import torch

from uavsec.baselines.deployment import heuristic_positions, min_enclosing_circle_center, polygon_centroid
from uavsec.channel import load_layout, reference_snr, sample_scenario
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


def test_swapped_layout_differs_only_in_eves_5_and_6():
    cfg = load_config([DEFAULT_CONFIG])
    base = load_layout("configs/layouts/paper_fig7.yaml", cfg.system, DEVICE)
    swapped = load_layout("configs/layouts/paper_fig7_swapped.yaml", cfg.system, DEVICE)
    assert torch.equal(base.users, swapped.users)
    same = [k for k in range(8) if k not in (5, 6)]
    assert torch.equal(base.eves[0, same], swapped.eves[0, same])
    assert torch.equal(base.eves[0, [5, 6]], swapped.eves[0, [6, 5]])


def test_eavesdropper_distance_spread():
    cfg = load_config([DEFAULT_CONFIG], ["system.eve_distance_spread=5.0", "system.area_size=10000"])
    scen = sample_scenario(cfg.system, 2000, make_generator(0, DEVICE), DEVICE)
    inside = ((scen.users > 25) & (scen.users < cfg.system.area_size - 25)).all(-1)   # not clipped at the border
    dist = (scen.users - scen.eves).norm(dim=-1)[inside]
    assert dist.min() >= 15.0 - 1e-3 and dist.max() <= 25.0 + 1e-3
    assert dist.std() > 2.0                                   # actually spread, not fixed
