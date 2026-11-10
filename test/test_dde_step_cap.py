import math

import pytest
import torch

from torchdde import AdaptiveStepSizeController, ConstantStepSizeController, integrate
from torchdde.solver import Bosh3, Dopri5, Euler, RK2, RK4


@pytest.mark.parametrize("solver_cls", [Bosh3, Dopri5])
@pytest.mark.parametrize("initial_dt", [0.05, 0.7, 2.0])
@pytest.mark.parametrize("duration", [0.1, 1.03])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("dto", [True, False])
def test_adaptive_delay_cap_preserves_solution_and_parameter_gradients(
    solver_cls, initial_dt, duration, dtype, dto,
):
    class Field(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rate = torch.nn.Parameter(torch.tensor(2.0, dtype=dtype))
            self.delays = torch.nn.Parameter(torch.tensor([0.37, 0.2], dtype=dtype))

        def forward(self, t, y, args, *, history):
            return self.rate.expand_as(y)

    model = Field()
    ts = torch.linspace(0.25, 0.25 + duration, 9, dtype=dtype)
    ys = integrate(
        model, solver_cls(), ts[0], ts[-1], ts,
        lambda t: torch.full((2, 2), 0.7, dtype=dtype), None,
        delays=model.delays, dt0=ts.new_tensor(initial_dt),
        stepsize_controller=AdaptiveStepSizeController(atol=1e-6, rtol=1e-3),
        discretize_then_optimize=dto,
    )
    expected = (0.7 + 2 * (ts - ts[0]))[None, :, None].expand_as(ys)
    torch.testing.assert_close(ys, expected)
    rate_grad, delay_grad = torch.autograd.grad(
        ys[:, -1].mean(), tuple(model.parameters()),
    )
    torch.testing.assert_close(rate_grad, ts[-1] - ts[0])
    torch.testing.assert_close(delay_grad, torch.zeros_like(model.delays))


@pytest.mark.parametrize("solver_cls", [Euler, RK2, RK4, Bosh3, Dopri5])
def test_constant_controller_caps_initial_step_and_final_endpoint(solver_cls):
    ts = torch.linspace(0, 1.03, 17, dtype=torch.float64)
    ys = integrate(
        lambda t, y, args, *, history: torch.ones_like(y),
        solver_cls(), ts[0], ts[-1], ts, lambda t: torch.zeros(1, 1, dtype=ts.dtype),
        None, delays=ts.new_tensor([0.2]), dt0=ts.new_tensor(0.7),
        stepsize_controller=ConstantStepSizeController(),
    )
    torch.testing.assert_close(ys[0, :, 0], ts)


@pytest.mark.parametrize("solver_cls", [Bosh3, Dopri5])
def test_adaptive_delay_cap_after_rejected_steps(solver_cls):
    class Controller(AdaptiveStepSizeController):
        def __init__(self):
            super().__init__(atol=1e-9, rtol=1e-7)
            self.rejections = 0

        def adapt_step_size(self, *args):
            result = super().adapt_step_size(*args)
            self.rejections += not bool(result[0])
            return result

    controller = Controller()
    ts = torch.linspace(0, 0.43, 17, dtype=torch.float64)
    rate = torch.tensor(4.0, dtype=ts.dtype, requires_grad=True)
    ys = integrate(
        lambda t, y, args, *, history: rate * y,
        solver_cls(), ts[0], ts[-1], ts, lambda t: torch.ones(1, 1, dtype=ts.dtype),
        None, delays=ts.new_tensor([0.2]), dt0=ts.new_tensor(0.7),
        stepsize_controller=controller,
    )
    assert controller.rejections > 0
    torch.testing.assert_close(
        ys[0, :, 0], (rate.detach() * ts).exp(), rtol=1e-5, atol=1e-7,
    )
    grad = torch.autograd.grad(ys[0, -1, 0], rate)[0]
    torch.testing.assert_close(
        grad, ts[-1] * (rate.detach() * ts[-1]).exp(), rtol=1e-5, atol=1e-7,
    )


@pytest.mark.parametrize("solver_cls", [Bosh3, Dopri5])
def test_delay_cap_preserves_history_dependent_solution(solver_cls):
    ts = torch.linspace(0, 1.03, 31, dtype=torch.float64)
    delay = 0.2
    ys = integrate(
        lambda t, y, args, *, history: history[0],
        solver_cls(), ts[0], ts[-1], ts, lambda t: torch.ones(1, 1, dtype=ts.dtype),
        None, delays=ts.new_tensor([delay]), dt0=ts.new_tensor(0.7),
        stepsize_controller=AdaptiveStepSizeController(atol=1e-9, rtol=1e-7),
    )
    # Method of steps for y'(t)=y(t-delay), with constant history y(t)=1.
    expected = torch.ones_like(ts)
    for power in range(1, 7):
        shifted = (ts - (power - 1) * delay).clamp_min(0)
        expected += shifted.pow(power) / math.factorial(power)
    torch.testing.assert_close(ys[0, :, 0], expected, rtol=1e-5, atol=1e-7)
