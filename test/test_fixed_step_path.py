import pytest
import torch

from torchdde import ConstantStepSizeController, integrate
from torchdde.solver import Bosh3, Dopri5, Euler, RK2, RK4


class ReferenceConstantController(ConstantStepSizeController):
    """Use the general integration path with identical constant-step semantics."""


@pytest.mark.parametrize("solver", [Euler(), RK2(), RK4(), Bosh3(), Dopri5()])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("problem", ["ode", "reverse_ode", "dde", "delay_boundary"])
def test_fixed_path_matches_general_values_and_parameter_gradients(solver, dtype, problem):
    dde = problem in ("dde", "delay_boundary")

    class Field(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rate = torch.nn.Parameter(torch.tensor(0.2, dtype=dtype))
            self.initial = torch.nn.Parameter(torch.ones(2, 2, dtype=dtype))
            first_delay = 0.07 if problem == "delay_boundary" else 0.17
            self.delays = torch.nn.Parameter(torch.tensor([first_delay, 0.23], dtype=dtype))

        def forward(self, t, y, args, *, history=None):
            result = self.rate * y
            if history is not None:
                result = result + 0.1 * history[0] + 0.05 * history[1]
            return result

    ts = torch.tensor([0.0, 0.03, 0.13, 0.21, 0.39, 0.4], dtype=dtype)
    dt = torch.tensor(0.07, dtype=dtype)
    if problem == "reverse_ode":
        ts, dt = ts.flip(0), -dt

    def run(controller):
        field = Field()
        initial = (lambda t: field.initial * t.exp()) if dde else field.initial
        ys = integrate(
            field, solver, ts[0], ts[-1], ts, initial, None,
            stepsize_controller=controller, dt0=dt,
            delays=field.delays if dde else None,
            discretize_then_optimize=True,
        )
        gradients = torch.autograd.grad(
            ys.square().sum(), tuple(field.parameters()), allow_unused=True,
        )
        return ys, gradients

    actual, gradients = run(ConstantStepSizeController())
    expected, reference_gradients = run(ReferenceConstantController())
    torch.testing.assert_close(actual, expected)
    for gradient, reference in zip(gradients, reference_gradients):
        if reference is None:
            assert gradient is None
        else:
            torch.testing.assert_close(gradient, reference)


def test_trainable_times_keep_general_path():
    ts = torch.tensor([0.0, 0.1, 0.2], dtype=torch.float64, requires_grad=True)

    def run(controller):
        value = integrate(
            lambda t, y, args: t * y, Euler(), ts[0], ts[-1], ts,
            torch.ones(1, 1, dtype=ts.dtype), None,
            stepsize_controller=controller, dt0=ts[1] - ts[0],
        )
        return torch.autograd.grad(value.sum(), ts)[0]

    torch.testing.assert_close(run(ConstantStepSizeController()), run(ReferenceConstantController()))
