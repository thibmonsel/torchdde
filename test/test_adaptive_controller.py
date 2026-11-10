import pytest
import torch

from torchdde import AdaptiveStepSizeController, Dopri5, integrate


def _solve(controller, device="cpu"):
    ts = torch.linspace(0, 2, 11, dtype=torch.float64, device=device)
    y0 = torch.tensor([[1.0, 0.5]], dtype=torch.float64, device=device)
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


def test_history_fills_slot_by_slot():
    controller = _pid_controller()
    errors = [torch.tensor(0.5), torch.tensor(0.25), torch.tensor(0.125)]
    controller.update_scaled_error(errors[0])
    assert controller.scaled_error == errors[0]
    assert not torch.isfinite(controller.prev_scaled_error)
    controller.update_scaled_error(errors[1])
    assert (controller.scaled_error, controller.prev_scaled_error) == tuple(errors[1::-1])
    assert not torch.isfinite(controller.prev_prev_scaled_error)
    controller.update_scaled_error(errors[2])
    assert (
        controller.scaled_error,
        controller.prev_scaled_error,
        controller.prev_prev_scaled_error,
    ) == tuple(errors[::-1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_solve_matches_cpu():
    cpu = _solve(_pid_controller())
    cuda = _solve(_pid_controller(), device="cuda")
    assert torch.allclose(cuda.cpu(), cpu, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("dtmin, dtmax", [(None, 0.05), (0.02, None), (0.01, 0.03)])
def test_step_size_bounds(dtmin, dtmax):
    controller = AdaptiveStepSizeController(atol=1e-6, rtol=1e-3, dtmin=dtmin, dtmax=dtmax)
    t0 = torch.tensor(0.0, dtype=torch.float64)
    y0 = torch.ones(1, 2, dtype=torch.float64)
    dt = torch.tensor(0.025, dtype=torch.float64)
    controller.init(None, t0, t0 + 1, y0, dt, None, 5)
    for y_error, expected in ((1e-12, dtmax), (1e3, dtmin)):
        *_, new_dt = controller.adapt_step_size(
            None, t0, t0 + dt, y0, y0, None, torch.full_like(y0, y_error), 5, dt
        )
        assert new_dt.dtype == torch.float64
        if expected is not None:
            assert torch.isclose(new_dt, torch.tensor(expected, dtype=torch.float64))
