"""Torch-dependent tests.  Skipped automatically when PyTorch is not installed."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spinenet_mo.evaluation.efficiency import measured_flops          # noqa: E402
from spinenet_mo.models.blocks import ECA, GhostConv                   # noqa: E402
from spinenet_mo.models.spinenet import SpineNet, count_parameters     # noqa: E402
from spinenet_mo.search import space                                   # noqa: E402
from spinenet_mo.search.analytic import count_spec                     # noqa: E402
from spinenet_mo.search.arch import build_spec                         # noqa: E402
from spinenet_mo.training.trainer import kl_to_reference               # noqa: E402

L = 4


def _random_genomes(n, seed=0):
    rng = np.random.default_rng(seed)
    return [space.random_genome(L, rng) for _ in range(n)]


@pytest.mark.parametrize("g", _random_genomes(8))
def test_forward_shape(g):
    model = SpineNet(build_spec(g, L)).eval()
    out = model(torch.randn(2, 1, 224, 224))
    assert out.shape == (2, 3) and torch.isfinite(out).all()


@pytest.mark.parametrize("g", _random_genomes(8, seed=1))
def test_analytic_params_equal_torch_params(g):
    spec = build_spec(g, L)
    assert count_spec(spec)["params"] == count_parameters(SpineNet(spec))


@pytest.mark.parametrize("g", _random_genomes(4, seed=2))
def test_analytic_flops_close_to_measured(g):
    spec = build_spec(g, L)
    analytic = 2 * count_spec(spec)["macs"]
    measured = measured_flops(SpineNet(spec))
    assert abs(analytic - measured) / measured < 0.1


def test_ghost_conv_output_channels_and_eca_shape():
    assert GhostConv(8, 20, 3, 1)(torch.randn(1, 8, 16, 16)).shape == (1, 20, 16, 16)
    x = torch.randn(2, 32, 8, 8)
    assert ECA(32)(x).shape == x.shape


def test_kl_loss_zero_for_matching_distributions():
    ref = torch.softmax(torch.randn(4, 3), 1)
    logits = torch.log(ref)
    assert torch.allclose(kl_to_reference(logits, ref), torch.zeros(4), atol=1e-5)


def test_one_training_step_reduces_loss():
    torch.manual_seed(0)
    model = SpineNet(build_spec(_random_genomes(1)[0], L))
    opt = torch.optim.AdamW(model.parameters(), 1e-3)
    x, y = torch.randn(8, 1, 224, 224), torch.randint(0, 3, (8,))
    first = None
    for _ in range(5):
        opt.zero_grad(); loss = torch.nn.functional.cross_entropy(model(x), y); loss.backward(); opt.step()
        first = first or loss.item()
    assert loss.item() < first
