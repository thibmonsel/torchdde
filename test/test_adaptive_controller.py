import torch

from torchdde import AdaptiveStepSizeController, Dopri5, integrate


def _solve(controller):
    ts = torch.linspace(0, 2, 11, dtype=torch.float64)
    y0 = torch.tensor([[1.0, 0.5]], dtype=torch.float64)
    return integrate(
        lambda t, y, args: torch.stack((y[:, 1], -4 * y[:, 0]), dim=-1),
        Dopri5(), ts[0], ts[-1], ts, y0, None,
        stepsize_controller=controller,
    )


def _pid_controller():
    return AdaptiveStepSizeController(
        atol=1e-8, rtol=1e-6, pcoeff=0.3, icoeff=0.3, dcoeff=0.1
    )


def test_reused_controller_matches_fresh_controller():
    controller = _pid_controller()
    first = _solve(controller)
    second = _solve(controller)
    assert torch.equal(first, second)
    assert torch.equal(second, _solve(_pid_controller()))


def test_rejected_step_keeps_accepted_error_history():
    controller = _pid_controller()
    t0 = torch.tensor(0.0, dtype=torch.float64)
    y0 = torch.ones(1, 2, dtype=torch.float64)
    controller.init(None, t0, t0 + 1, y0, torch.tensor(0.1, dtype=torch.float64),
                    None, 5)
    args = (None, t0, t0 + 0.1, y0, y0, None)
    controller.adapt_step_size(*args, torch.full_like(y0, 1e-9), 5,
                               torch.tensor(0.1, dtype=torch.float64))
    accepted = float(controller.scaled_error)
    controller.adapt_step_size(*args, torch.full_like(y0, 1.0), 5,
                               torch.tensor(0.1, dtype=torch.float64))
    assert float(controller.scaled_error) == accepted
    assert not torch.isfinite(controller.prev_scaled_error)
