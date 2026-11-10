import pytest
import torch

from torchdde import integrate
from torchdde.solver import Bosh3, Dopri5, Euler, RK2, RK4


@pytest.mark.parametrize("solver_cls", [Euler, RK2, RK4, Bosh3, Dopri5])
@pytest.mark.parametrize("fixed_kind", ["buffer", "frozen_parameter"])
@pytest.mark.parametrize("delays", [[0.173], [0.271, 0.113, 0.173, 0.223]])
def test_fixed_delay_network_gradients_match_trainable_delay_reference(
    solver_cls, fixed_kind, delays,
):
    dtype = torch.float64

    class Field(torch.nn.Module):
        def __init__(self, kind):
            super().__init__()
            values = torch.tensor(delays, dtype=dtype)
            if kind == "buffer":
                self.register_buffer("delays", values)
            else:
                self.delays = torch.nn.Parameter(values, requires_grad=kind == "trainable")
            self.rate = torch.nn.Parameter(torch.tensor(0.15, dtype=dtype))
            self.unused = torch.nn.Parameter(torch.tensor(2.0, dtype=dtype))

        def forward(self, t, y, args, *, history):
            combined = sum(0.1 / (i + 1) * h for i, h in enumerate(history))
            return self.rate * y + y * combined.sin()

    ts = torch.linspace(0.25, 0.65, 9, dtype=dtype)

    def run(kind):
        model = Field(kind)
        ys = integrate(
            model, solver_cls(), ts[0], ts[-1], ts,
            lambda t: torch.ones(2, 2, dtype=dtype) * t.exp(), None,
            delays=model.delays, dt0=ts[1] - ts[0],
            discretize_then_optimize=False,
        )
        ys.square().mean().backward()
        return ys, model

    actual, fixed = run(fixed_kind)
    reference, trainable = run("trainable")
    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(fixed.rate.grad, trainable.rate.grad)
    torch.testing.assert_close(fixed.unused.grad, torch.zeros_like(fixed.unused))
    assert fixed.delays.grad is None


def test_fixed_delays_do_not_require_history_time_derivatives():
    class Field(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("delays", torch.tensor([1.2]))
            self.rate = torch.nn.Parameter(torch.tensor(0.15))

        def forward(self, t, y, args, *, history):
            return self.rate * y + 0.1 * history[0]

    def history(t):
        # Fixed-delay training must not request a JVP of the history function.
        assert not t.requires_grad
        return t.exp().reshape(1, 1)

    model = Field()
    ts = torch.linspace(0.0, 0.4, 9)
    ys = integrate(
        model, Dopri5(), ts[0], ts[-1], ts, history, None,
        delays=model.delays, dt0=ts[1], discretize_then_optimize=False,
    )
    ys[:, -1].sum().backward()
    assert model.rate.grad is not None
    assert torch.isfinite(model.rate.grad)
