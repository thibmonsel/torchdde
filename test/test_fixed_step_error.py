import pytest
import torch

from torchdde import AdaptiveStepSizeController, ConstantStepSizeController, integrate
from torchdde.solver import Bosh3, Dopri5, Euler, RK2, RK4


@pytest.mark.parametrize("solver_cls", [RK2, RK4, Bosh3, Dopri5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fixed_runge_kutta_step_preserves_values_dense_data_and_gradients(solver_cls, dtype):
    def run(fixed):
        solver = solver_cls()
        y0 = torch.ones(2, 3, dtype=dtype, requires_grad=True)
        rate = torch.tensor(0.7, dtype=dtype, requires_grad=True)
        t = torch.tensor(0.2, dtype=dtype, requires_grad=True)
        dt = torch.tensor(0.1, dtype=dtype, requires_grad=True)
        func = lambda time, y, args: rate * y + time
        state = solver.init(func, t, y0, dt, None)
        method = solver.step_fixed if fixed else solver.step
        y1, error, dense, state, aux = method(func, t, y0, dt, state, None)
        interp = solver.build_interpolation(t, t + dt, dense)
        loss = y1.square().sum() + interp(t + dt / 2).square().sum()
        gradients = torch.autograd.grad(loss, (y0, rate, t, dt))
        return y1, error, dense, state, aux, gradients

    actual, error, dense, state, aux, gradients = run(True)
    expected, reference_error, reference_dense, reference_state, reference_aux, reference_gradients = run(False)
    assert error is None
    assert isinstance(reference_error, torch.Tensor)
    assert aux is reference_aux is None
    torch.testing.assert_close(actual, expected)
    for key in dense:
        torch.testing.assert_close(dense[key], reference_dense[key])
    if reference_state is None:
        assert state is None
    else:
        for actual_state, expected_state in zip(state, reference_state):
            torch.testing.assert_close(actual_state, expected_state)
    for actual_gradient, expected_gradient in zip(gradients, reference_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("base", [Euler, RK2])
def test_fixed_integration_preserves_custom_step_overrides(base):
    class CustomSolver(base):
        calls = 0

        def step(self, func, t, y, dt, state, args):
            self.calls += 1
            wrapped = lambda time, value, extra: func(time, value, extra) + 1.0
            return super().step(wrapped, t, y, dt, state, args)

    solver = CustomSolver()
    t = torch.tensor(0.0)
    state = solver.init(lambda t, y, args: y, t, torch.ones(1, 1), t + 0.1, None)
    result = solver.step_fixed(lambda t, y, args: torch.zeros_like(y), t,
                               torch.ones(1, 1), t + 0.1, state, None)
    assert solver.calls == 1
    torch.testing.assert_close(result[0], torch.tensor([[1.1]]))
    ts = torch.tensor([0.0, 0.1, 0.2])
    ys = integrate(lambda t, y, args: torch.zeros_like(y), solver,
                   ts[0], ts[-1], ts, torch.ones(1, 1), None, dt0=ts[1])
    assert solver.calls == 3
    torch.testing.assert_close(ys[:, -1], torch.tensor([[1.2]]))


@pytest.mark.parametrize("adaptive", [False, True])
def test_nonstandard_controllers_still_receive_error_estimates(adaptive):
    base = AdaptiveStepSizeController if adaptive else ConstantStepSizeController

    class Controller(base):
        checks = 0

        def adapt_step_size(self, func, t0, t1, y0, y1, args, y_error, order, dt):
            assert isinstance(y_error, torch.Tensor)
            self.checks += 1
            return super().adapt_step_size(func, t0, t1, y0, y1, args, y_error, order, dt)

    controller = Controller(atol=1e-6, rtol=1e-4) if adaptive else Controller()
    ts = torch.tensor([0.0, 0.1, 0.2])
    result = integrate(lambda t, y, args: y, Dopri5(), ts[0], ts[-1], ts,
                       torch.ones(1, 1), None, dt0=ts[1], stepsize_controller=controller)
    assert controller.checks > 0
    torch.testing.assert_close(result[:, -1], torch.exp(ts[-1]).reshape(1, 1))
