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


@pytest.mark.parametrize("out_layers,init", [(3, "zeros"), (2, "mrt")])
def test_gnn_variants_stay_permutation_equivariant(cfg, out_layers, init):
    torch.manual_seed(0)
    model = SecureBeamGNN(cfg.system.num_antennas, reference_snr(cfg.system), hidden_dim=16, num_layers=2,
                          out_layers=out_layers, init_embedding=init)
    h, h_e = batch(cfg, k=5)
    perm = torch.randperm(5)
    assert torch.allclose(model(h, h_e, 1.0)[:, perm], model(h[:, perm], h_e[:, perm], 1.0), atol=1e-5)


def test_scorecard_relative_error():
    from uavsec.scorecard import relative_error
    assert relative_error([10.0, 0.5], [10.0, 0.0]) == pytest.approx(0.25)   # floor of 1 for tiny paper values


def test_beamforming_parts_need_the_right_mlps():
    import importlib.util, sys
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("exp_beamforming", Path(__file__).parents[1] / "scripts" / "exp_beamforming.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    ev = load_config([DEFAULT_CONFIG]).eval
    assert mod.needed_mlp_users({"power"}, ev, 8) == {8}
    assert mod.needed_mlp_users({"users"}, ev, 8) == set(ev.sweep_users)
    assert mod.needed_mlp_users({"cdf", "noise"}, ev, 8) == set(ev.cdf_users) | {8}


def test_empty_aggregate_only_changes_k1(cfg):
    model = make_gnn(cfg)
    h, h_e = batch(cfg, k=4)
    w_zero = model(h, h_e, 1.0)
    h1, he1 = batch(cfg, k=1)
    w1_zero = model(h1, he1, 1.0)
    model.set_empty_aggregate("self")
    assert torch.allclose(model(h, h_e, 1.0), w_zero)            # K >= 2: identical
    assert not torch.allclose(model(h1, he1, 1.0), w1_zero)      # K = 1: rule applies
    perm = torch.randperm(4)
    assert torch.allclose(model(h, h_e, 1.0)[:, perm], model(h[:, perm], h_e[:, perm], 1.0), atol=1e-5)
    with pytest.raises(ValueError):
        model.set_empty_aggregate("mean")
