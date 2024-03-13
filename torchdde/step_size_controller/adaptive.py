import torch

from .base import AbstractStepSizeController


def rms_norm(y):
    return y.abs().pow(2).mean().sqrt()


def max_norm(y):
    return y.abs().max()


def _select_initial_step(func, t0, y0, args, order, rtol, atol, norm, f0=None):
    """Empirically select a good initial step.

    The algorithm is described in [1]_.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.4, 2nd edition.
    """

    dtype = y0.dtype
    device = y0.device
    t_dtype = t0.dtype
    t0 = t0.to(t_dtype)

    if f0 is None:
        f0 = func(t0, y0, args)

    scale = atol + torch.abs(y0) * rtol

    d0 = norm(y0 / scale).abs()
    d1 = norm(f0 / scale).abs()

    if d0 < 1e-5 or d1 < 1e-5:
        h0 = torch.tensor(1e-6, dtype=dtype, device=device)
    else:
        h0 = 0.01 * d0 / d1
    h0 = h0.abs()

    y1 = y0 + h0 * f0
    f1 = func(t0 + h0, y1)

    d2 = torch.abs(norm((f1 - f0) / scale) / h0)

    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = torch.max(torch.tensor(1e-6, dtype=dtype, device=device), h0 * 1e-3)
    else:
        h1 = (torch.max(d1, d2) * 100).reciprocal().pow(1.0 / float(order + 1))
    h1 = h1.abs()

    return torch.min(100 * h0, h1).to(t_dtype)


def _compute_error_ratio(error_estimate, rtol, atol, y0, y1, norm):
    error_tol = atol + rtol * torch.max(y0.abs(), y1.abs())
    return norm(error_estimate / error_tol).abs()


@torch.no_grad()
def _optimal_step_size(last_step, error_ratio, safety, icoeff, dcoeff, order):
    """Calculate the optimal size for the next step."""
    if error_ratio == 0:
        return last_step * icoeff
    if error_ratio < 1:
        dcoeff = torch.ones((), dtype=last_step.dtype, device=last_step.device)
    error_ratio = error_ratio.type_as(last_step)
    exponent = torch.tensor(
        order, dtype=last_step.dtype, device=last_step.device
    ).reciprocal()
    factor = torch.min(
        torch.tensor(icoeff), torch.max(safety / error_ratio**exponent, dcoeff)
    )
    return last_step * factor


class AdaptiveStepSizeController(AbstractStepSizeController):
    def __init__(self, atol, rtol, safety=0.9, icoeff=10.0, dcoeff=0.2) -> None:
        super().__init__()
        self.atol = atol
        self.rtol = rtol
        self.safety = safety
        self.icoeff = icoeff
        self.dcoeff = dcoeff

    def init(self, func, t0, t1, y0, dt0, args, error_order):
        del t1
        if dt0 is None:
            dt0 = _select_initial_step(
                func, t0, y0, args, error_order, self.rtol, self.atol, rms_norm
            )
            return t0 + dt0, dt0
        else:
            return t0 + dt0, dt0

    def adapt_step_size(
        self,
        func,
        t0,
        t1,
        y0,
        y1_candidate,
        args,
        y_error,
        error_order,
        controller_state,
    ):
        del func, args
        error_ratio = _compute_error_ratio(
            y_error, self.rtol, self.atol, y0, y1_candidate, rms_norm
        )
        keep_step = error_ratio <= 1
        next_controller_state = _optimal_step_size(
            controller_state,
            error_ratio,
            self.safety,
            self.icoeff,
            self.dcoeff,
            error_order,
        )
        next_t0 = t1 if keep_step else t0
        next_t1 = (
            t1 + next_controller_state if keep_step else t0 + next_controller_state
        )
        return keep_step, next_t0, next_t1, next_controller_state
