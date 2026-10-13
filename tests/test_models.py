"""Structural properties of the learned beamformers."""
import pytest
import torch

from uavsec.channel import area_center, channels, reference_snr, sample_scenario
from uavsec.config import DEFAULT_CONFIG, load_config
from uavsec.models import MLPBeamformer, SecureBeamGNN, load_model, save_model
from uavsec.models.gnn import max_over_others
from uavsec.utils import make_generator

DEVICE = torch.device("cpu")


@pytest.fixture(scope="module")
def cfg():
    return load_config([DEFAULT_CONFIG])


def make_gnn(cfg):
    torch.manual_seed(0)
    return SecureBeamGNN(cfg.system.num_antennas, reference_snr(cfg.system),
                         hidden_dim=32, num_layers=2)


def batch(cfg, k, b=3, seed=0):
    gen = make_generator(seed, DEVICE)
    scen = sample_scenario(cfg.system, b, gen, DEVICE, num_users=k)
    return channels(cfg.system, area_center(cfg.system, b, DEVICE), scen, gen)


def test_max_over_others():
    x = torch.tensor([[[1.0], [5.0], [3.0]]])
    assert max_over_others(x).flatten().tolist() == [5.0, 3.0, 5.0]
    assert torch.all(max_over_others(x[:, :1]) == 0)


def test_gnn_is_permutation_equivariant(cfg):
    model = make_gnn(cfg)
    h, h_e = batch(cfg, k=6)
    perm = torch.randperm(6)
    w = model(h, h_e, cfg.system.power_max_w)
    w_perm = model(h[:, perm], h_e[:, perm], cfg.system.power_max_w)
    assert torch.allclose(w[:, perm], w_perm, atol=1e-5)


@pytest.mark.parametrize("k", [1, 4, 12])
def test_gnn_handles_any_k_and_meets_power(cfg, k):
    model = make_gnn(cfg)
    h, h_e = batch(cfg, k)
    w = model(h, h_e, 0.5)
    assert w.shape == h.shape
    assert torch.allclose(w.abs().square().sum(dim=(1, 2)), torch.full((3,), 0.5), rtol=1e-5)


def test_mlp_rejects_wrong_k(cfg):
    model = MLPBeamformer(cfg.system.num_antennas, reference_snr(cfg.system), num_users=4, hidden_dims=[16])
    h, h_e = batch(cfg, k=4)
    assert model(h, h_e, 1.0).shape == h.shape
    with pytest.raises(ValueError):
        model(*batch(cfg, k=5), 1.0)


def test_checkpoint_roundtrip(cfg, tmp_path):
    model = make_gnn(cfg)
    save_model(model, tmp_path / "m.pt")
    loaded = load_model(tmp_path / "m.pt")
    h, h_e = batch(cfg, k=3)
    assert torch.allclose(model.eval()(h, h_e, 1.0), loaded(h, h_e, 1.0))
