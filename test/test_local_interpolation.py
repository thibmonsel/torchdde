import torch
import pytest
from torchdde import ThirdOrderPolynomialInterpolation
from torchdde.solver import Dopri5, RK4
from torchdde.local_interpolation.fourth_order_interpolation import linear_rescale


@pytest.mark.parametrize("solver, degree", [(RK4(), 3), (Dopri5(), 4)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_horner_values_and_gradients_match_power_basis(solver, degree, dtype):
    generator = torch.Generator().manual_seed(3)
    coeffs = torch.rand(2, degree + 1, 3, dtype=dtype, generator=generator, requires_grad=True)
    start = torch.tensor(-0.2, dtype=dtype, requires_grad=True)
    end = torch.tensor(0.7, dtype=dtype, requires_grad=True)
    query = torch.tensor(0.37, dtype=dtype, requires_grad=True)
    interpolator = solver.interpolation_cls(
        start, end, {"_cached_coefficients": coeffs},
    )
    actual = interpolator(query)
    normalized = (query - start) / (end - start)
    powers = normalized ** torch.arange(degree, -1, -1, dtype=dtype)
    expected = torch.einsum("c,bcf->bf", powers, coeffs)
    torch.testing.assert_close(actual, expected)
    inputs = (coeffs, start, end, query)
    actual_grads = torch.autograd.grad(actual.square().sum(), inputs, retain_graph=True)
    expected_grads = torch.autograd.grad(expected.square().sum(), inputs)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_linear_rescale_degenerate_interval_has_finite_gradients():
    t = torch.tensor(2.0, requires_grad=True)
    result = linear_rescale(torch.tensor(2.0), t, torch.tensor(2.0))
    result.backward()
    torch.testing.assert_close(result, torch.tensor(0.0))
    torch.testing.assert_close(t.grad, torch.tensor(0.0))


def test_linear_rescale_value_and_gradient():
    t = torch.tensor(2.25, requires_grad=True)
    result = linear_rescale(torch.tensor(2.0), t, torch.tensor(2.5))
    result.backward()
    torch.testing.assert_close(result, torch.tensor(0.5))
    torch.testing.assert_close(t.grad, torch.tensor(2.0))


def test_third_order_interpolation():
    t0 = torch.tensor(2.0)
    t1 = torch.tensor(3.9)

    def y(t):
        return 0.4 + 0.7 * t - 1.1 * t**2 + 0.4 * t**3

    def yprime(t):
        return 0.7 - 2.2 * t + 1.2 * t**2

    y0, f0 = y(t0), yprime(t0)
    y1, f1 = y(t1), yprime(t1)
    k0 = f0
    k1 = f1
    y0, f0, y1, f1, k0, k1 = (
        y0.reshape(-1, 1),
        f0.reshape(-1, 1),
        y1.reshape(-1, 1),
        f1.reshape(-1, 1),
        k0.reshape(-1, 1),
        k1.reshape(-1, 1),
    )
    dense_info = dict(y0=y0, y1=y1, k=torch.stack([k0, k1]))
    interp = ThirdOrderPolynomialInterpolation(t0=t0, t1=t1, dense_info=dense_info)
    assert torch.allclose(
        y(torch.tensor(2.6)).flatten(),
        interp(torch.tensor(2.6)).flatten(),
    )
