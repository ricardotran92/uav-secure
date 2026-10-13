"""Checks for the channel model and the rate expressions (eqs. (4)-(6), (19), (21)-(22))."""
import math

import numpy as np
import pytest
import torch

from uavsec.channel import (area_center, channels, path_gain, reference_snr, sample_scenario,
                            steering_vector)
from uavsec.config import DEFAULT_CONFIG, load_config
from uavsec.secrecy import (normalize_power, project_power_ball, secrecy_rates, sinr_rates,
                            sum_secrecy_rate)
from uavsec.utils import make_generator

DEVICE = torch.device("cpu")


@pytest.fixture(scope="module")
def sys_cfg():
    return load_config([DEFAULT_CONFIG]).system


def random_instance(sys_cfg, batch=4, seed=0):
    gen = make_generator(seed, DEVICE)
    scen = sample_scenario(sys_cfg, batch, gen, DEVICE)
    h, h_e = channels(sys_cfg, area_center(sys_cfg, batch, DEVICE), scen, gen)
    w_raw = torch.complex(torch.randn(h.shape, generator=gen), torch.randn(h.shape, generator=gen))
    return scen, h, h_e, normalize_power(w_raw, sys_cfg.power_max_w)


def test_eavesdroppers_near_users(sys_cfg):
    scen = sample_scenario(sys_cfg, 256, make_generator(1, DEVICE), DEVICE)
    dist = (scen.users - scen.eves).norm(dim=-1)
    assert torch.all(dist <= sys_cfg.eve_distance + 1e-4)
    assert torch.all((scen.eves >= 0) & (scen.eves <= sys_cfg.area_size))
    assert dist.median() == pytest.approx(sys_cfg.eve_distance, rel=1e-4)  # most are not clipped


def test_channel_average_gain_matches_path_loss(sys_cfg):
    """E|h_n|^2 = beta(d) / sigma^2 for a unit-modulus LoS + unit-variance NLoS mixture."""
    gen = make_generator(2, DEVICE)
    batch = 4000
    users = torch.full((batch, 1, 2), 100.0)
    scen = sample_scenario(sys_cfg, batch, gen, DEVICE, num_users=1)
    scen.users = users
    h, _ = channels(sys_cfg, area_center(sys_cfg, batch, DEVICE), scen, gen)
    expected = path_gain(sys_cfg, torch.tensor(sys_cfg.uav_altitude)) / sys_cfg.noise_power_w
    assert h.abs().square().mean().item() == pytest.approx(expected.item(), rel=0.03)
    assert reference_snr(sys_cfg) == pytest.approx(sys_cfg.power_max_w * expected.item(), rel=1e-5)


def test_steering_vector_unit_modulus():
    a = steering_vector(torch.tensor([0.3, -0.7]), 8, 0.5)
    assert a.shape == (2, 8)
    assert torch.allclose(a.abs(), torch.ones(2, 8))


def test_rates_match_numpy_reference(sys_cfg):
    _, h, h_e, w = random_instance(sys_cfg)
    ours = secrecy_rates(h, h_e, w, clamp=False).numpy()
    hn, hen, wn = h.numpy().astype(complex), h_e.numpy().astype(complex), w.numpy().astype(complex)
    for b in range(h.shape[0]):
        for k in range(h.shape[1]):
            def rate(ch):
                sig = abs(np.vdot(ch[b, k], wn[b, k])) ** 2
                inter = sum(abs(np.vdot(ch[b, k], wn[b, l])) ** 2 for l in range(h.shape[1]) if l != k)
                return math.log2(1 + sig / (inter + 1.0))
            assert ours[b, k] == pytest.approx(rate(hn) - rate(hen), rel=1e-3, abs=1e-3)


def test_real_domain_form_eq21_matches_complex(sys_cfg):
    """Eq. (21): |h^H w|^2 = ||[[Re h, Im h], [-Im h, Re h]]^T-style real product||^2."""
    _, h, _, w = random_instance(sys_cfg, batch=1)
    hk, wl = h[0, 0], w[0, 1]
    upper = torch.dot(hk.real, wl.real) + torch.dot(hk.imag, wl.imag)   # Re(h^H w)
    lower = torch.dot(hk.real, wl.imag) - torch.dot(hk.imag, wl.real)   # Im(h^H w)
    gamma = torch.stack((upper, lower))
    assert gamma.square().sum().item() == pytest.approx(torch.vdot(hk, wl).abs().square().item(), rel=1e-4)


def test_power_normalisation_and_projection(sys_cfg):
    _, h, h_e, w = random_instance(sys_cfg)
    power = w.abs().square().sum(dim=(1, 2))
    assert torch.allclose(power, torch.full_like(power, sys_cfg.power_max_w), rtol=1e-5)
    small = project_power_ball(0.1 * w, sys_cfg.power_max_w)
    assert torch.allclose(small, 0.1 * w)                       # feasible: unchanged
    big = project_power_ball(10 * w, sys_cfg.power_max_w)
    assert torch.allclose(big.abs().square().sum(dim=(1, 2)), power, rtol=1e-5)


def test_secrecy_is_nonnegative_and_sums(sys_cfg):
    _, h, h_e, w = random_instance(sys_cfg)
    per_user = secrecy_rates(h, h_e, w)
    assert torch.all(per_user >= 0)
    assert torch.allclose(per_user.sum(-1), sum_secrecy_rate(h, h_e, w))


def test_single_user_mrt_is_snr_limited(sys_cfg):
    """K=1, MRT: R = log2(1 + P ||h||^2) (no interference)."""
    _, h, _, _ = random_instance(sys_cfg)
    h1 = h[:, :1]
    w = normalize_power(h1, sys_cfg.power_max_w)
    expected = torch.log2(1 + sys_cfg.power_max_w * h1.abs().square().sum(dim=(1, 2)))
    assert torch.allclose(sinr_rates(h1, w)[:, 0], expected, rtol=1e-4)
