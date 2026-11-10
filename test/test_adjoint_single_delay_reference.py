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
    [[0.30982711472371577, 0.030173444859053968],
     [0.08215017649869406, 0.009076283440657883]],
    [[0.3189182875115662, -0.008932560559085359],
     [0.08484184128560227, -0.0007147029116144612]],
]
DELAY_GRAD = [0.03132345020716109]


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
