import pytest
import torch

from torchdde.local_interpolation import FirstOrderPolynomialInterpolation
from torchdde.solver.runge_kutta import ButcherTableau, ExplicitRungeKutta


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("coefficient", [-0.2, 0.0, 0.3, 1.0])
def test_first_stage_preserves_custom_tableau_values_and_gradients(
    dtype, coefficient,
):
    class CustomSolver(ExplicitRungeKutta):
        def order(self):
            return 1

    tableau = ButcherTableau.from_lists(
        c=[0.0, 0.4], a=[[], [coefficient]], b=[0.25, 0.75],
        b_err=[0.1, -0.1], dtype=dtype,
    )
    solver = CustomSolver(tableau, FirstOrderPolynomialInterpolation)
    y0 = torch.tensor([[1.0, 2.0]], dtype=dtype, requires_grad=True)
    rate = torch.tensor(0.7, dtype=dtype, requires_grad=True)
    time = torch.tensor(0.2, dtype=dtype, requires_grad=True)
    dt = torch.tensor(0.1, dtype=dtype, requires_grad=True)
    func = lambda t, y, args: rate * y + t
    state = solver.init(func, time, y0, dt, None)
    weights = solver.tableau.a.requires_grad_()
    actual, error, dense, _, _ = solver.step(func, time, y0, dt, state, None)

    k0 = rate * y0 + time
    stage_state = y0 + dt * weights[1, 0] * k0
    k1 = rate * stage_state + time + dt * solver.tableau.c[1]
    expected = y0 + dt * (solver.tableau.b[0] * k0 + solver.tableau.b[1] * k1)
    expected_error = dt * (solver.tableau.b_err[0] * k0 + solver.tableau.b_err[1] * k1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(error, expected_error)
    torch.testing.assert_close(dense["k"], torch.stack((k0, k1)))
    loss = actual.square().sum() + error.square().sum() + dense["k"].square().sum()
    reference_loss = (
        expected.square().sum() + expected_error.square().sum()
        + k0.square().sum() + k1.square().sum()
    )
    inputs = (y0, rate, time, dt, weights)
    gradients = torch.autograd.grad(loss, inputs, retain_graph=True)
    reference_gradients = torch.autograd.grad(reference_loss, inputs)
    for gradient, reference in zip(gradients, reference_gradients):
        torch.testing.assert_close(gradient, reference)
