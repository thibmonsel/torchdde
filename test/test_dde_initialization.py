import pytest
import torch

from torchdde import AdaptiveStepSizeController, integrate
from torchdde.solver import Bosh3, Dopri5


@pytest.mark.parametrize("solver", [Dopri5(), Bosh3()])
@pytest.mark.parametrize("automatic_step", [False, True])
def test_dde_initialization_can_evaluate_history(solver, automatic_step):
    ts = torch.linspace(0, 0.25, 11, dtype=torch.float64)
    initial = torch.ones((2, 1), dtype=torch.float64)

    def history(t):
        assert t <= ts[0]
        return initial

    ys = integrate(
        lambda t, y, args, history: -history[0],
        solver,
        ts[0],
        ts[-1],
        ts,
        history,
        None,
        stepsize_controller=AdaptiveStepSizeController(atol=1e-8, rtol=1e-6),
        delays=torch.tensor([0.5], dtype=torch.float64),
        dt0=None if automatic_step else ts[1] - ts[0],
        discretize_then_optimize=True,
    )

    torch.testing.assert_close(
        ys, initial[:, None] * (1 - ts)[None, :, None], rtol=1e-6, atol=1e-8
    )
