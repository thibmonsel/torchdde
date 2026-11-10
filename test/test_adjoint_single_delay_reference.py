"""Pin single-delay adjoint gradients, to catch unintended numerical changes.

Regenerate the values only for intentional accuracy changes to the adjoint.

The delay (0.3) is a multiple of the step (0.1), so the history boundary
t - tau == t0 is hit exactly, covering both lookup branches.
"""
import pytest
import torch
from torch import nn

from torchdde import RK4, integrate

PARAM_GRADS = [
    [[0.3101451353326353, 0.030167044228006083],
     [0.08197887203929763, 0.0090712422141648]],
    [[0.31923802391118217, -0.008992427843783013],
     [0.08466804489642865, -0.0006935351191357376]],
]
DELAY_GRAD = [0.030977314048086357]


class Model(nn.Module):
    def __init__(self, learn_delay):
        super().__init__()
        self.delays = nn.Parameter(
            torch.tensor([0.3], dtype=torch.float64), requires_grad=learn_delay
        )
        self.weight = nn.Parameter(
            torch.tensor([[-0.7, 0.2], [0.1, -0.4]], dtype=torch.float64)
        )
        self.mix = nn.Parameter(
            torch.tensor([[0.5, -0.3], [0.2, 0.6]], dtype=torch.float64)
        )

    def forward(self, t, y, args, *, history):
        return torch.tanh(y @ self.weight.T + history[0] @ self.mix.T)


@pytest.mark.parametrize("learn_delay", [True, False])
def test_single_delay_adjoint_gradients_match_reference(learn_delay):
    model = Model(learn_delay)
    ts = torch.linspace(0, 1.2, 13, dtype=torch.float64)
    scale = torch.tensor([[1.0], [0.5], [-1.0]], dtype=torch.float64)
    ys = integrate(
        model, RK4(), ts[0], ts[-1], ts,
        lambda t: torch.stack((torch.cos(t), torch.sin(t))).expand(3, 2) * scale,
        None, dt0=ts[1] - ts[0], delays=model.delays,
        discretize_then_optimize=False,
    )
    ys.square().mean().backward()
    expected = ([DELAY_GRAD] if learn_delay else []) + PARAM_GRADS
    actual = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(actual) == len(expected)
    for grad, ref in zip(actual, expected):
        torch.testing.assert_close(
            grad, torch.tensor(ref, dtype=torch.float64), rtol=1e-12, atol=1e-14
        )
