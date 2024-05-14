import torch

from .base import AbstractStepSizeController


def rms_norm(y):
    return y.abs().pow(2).mean().sqrt()


def max_norm(y):
    return y.abs().max()


def _select_initial_step(func, t0, y0, args, order, rtol, atol, norm, f0=None):
    """
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
    f1 = func(t0 + h0, y1, args)

    d2 = torch.abs(norm((f1 - f0) / scale) / h0)

    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = torch.max(torch.tensor(1e-6, dtype=dtype, device=device), h0 * 1e-3)
    else:
        h1 = (torch.max(d1, d2) * 100).reciprocal().pow(1.0 / float(order + 1))
    h1 = h1.abs()

    return torch.min(100 * h0, h1).to(t_dtype)


@torch.no_grad()
def _optimal_step_size_with_pi(last_step, scaled_error, safety, icoeff, dcoeff, order):
    """Calculate the optimal size for the next step."""
    if scaled_error == torch.zeros_like(scaled_error):
        return last_step * icoeff
    if scaled_error < 1:
        dcoeff = torch.ones((), dtype=last_step.dtype, device=last_step.device)
    exponent = torch.tensor(
        order, dtype=last_step.dtype, device=last_step.device
    ).reciprocal()
    factor = torch.min(
        torch.tensor(icoeff),
        torch.max(safety / scaled_error**exponent, torch.tensor(dcoeff)),
    )
    return last_step * factor


@torch.no_grad()
def _optimal_step_size_with_pid(
    last_dt,
    scaled_errors,
    safety,
    pcoeff,
    icoeff,
    dcoeff,
    order,
    factormin=0.2,
    factormax=10.0,
):
    """Calculate the optimal size for the next step.
    Some could perform
    h_{n+1} = δ_{n,n}^β_1 * δ_{n-1,n-1}^β_2 * δ_{n-2,n-2}^β_3 * h_n
    where
    h_n is the nth step size
    ε_n     = atol + norm(y) * rtol with y on the nth step
    r_n     = norm(y_error) with y_error on the nth step
    δ_{n,m} = norm(y_error / (atol + norm(y) * rtol))^(-1) with y_error on the nth
    step and y on the mth step
    β_1     = pcoeff + icoeff + dcoeff
    β_2     = -(pcoeff + 2 * dcoeff)
    β_3     = dcoeff
    """
    beta1 = (pcoeff + icoeff + dcoeff) / order
    beta2 = -(pcoeff + 2 * dcoeff) / order
    beta3 = dcoeff / order
    zero_coeff_or_inv_error = (
        lambda coeff, inv_error: coeff == 0 or torch.zeros_like(inv_error) == inv_error
    )
    scaled_error, prev_scaled_errors, prev_prev_scaled_errors = scaled_errors

    inv_scaled_error, inv_prev_scaled_errors, inv_prev_prev_scaled_errors = (
        scaled_error.reciprocal(),
        prev_scaled_errors.reciprocal(),
        prev_prev_scaled_errors.reciprocal(),
    )
    factor1 = (
        torch.tensor(1)
        if zero_coeff_or_inv_error(beta1, inv_scaled_error)
        else inv_scaled_error**beta1
    )
    factor2 = (
        torch.tensor(1)
        if zero_coeff_or_inv_error(beta2, inv_prev_scaled_errors)
        else inv_prev_scaled_errors**beta2
    )
    factor3 = (
        torch.tensor(1)
        if zero_coeff_or_inv_error(beta3, inv_prev_prev_scaled_errors)
        else inv_prev_prev_scaled_errors**beta3
    )
    factor = torch.clip(
        safety * factor1 * factor2 * factor3,
        factormin,
        factormax,
    )
    return last_dt * factor


class AdaptiveStepSizeController(AbstractStepSizeController):
    """Adapts the step size to produce a solution accurate to a given tolerance.
    The tolerance is calculated as `atol + rtol * y` for the evolving solution `y`.

    Steps are adapted using a PID controller.

    ??? tip "Choosing tolerances"

        The choice of `rtol` and `atol` are used to determine how accurately you would
        like the numerical approximation to your equation. If you are solving a problem
        "harder" problem then you probably need to raise the tolerances to get an
        appropriate solution.

        Default values usually are `rtol=1e-3` and `atol=1e-6`.

    ??? tip "Choosing PID coefficients"

        We refer the reader to `Diffrax` clear explanation
        [here](https://docs.kidger.site/diffrax/api/stepsize_controller/).
    """

    def __init__(
        self,
        atol,
        rtol,
        safety=0.9,
        pcoeff=0.0,
        icoeff=1.0,
        dcoeff=0.0,
        factormax=10.0,
        dtmin=None,
        dtmax=None,
    ) -> None:
        super().__init__()
        self.atol = atol
        self.rtol = rtol
        self.safety = safety
        self.pcoeff = pcoeff
        self.icoeff = icoeff
        self.dcoeff = dcoeff
        self.factormax = factormax
        self.dtmin = dtmin
        self.dtmax = dtmax
        self.scaled_error = torch.inf * torch.tensor((1.0))
        self.prev_scaled_error = torch.inf * torch.tensor((1.0))
        self.prev_prev_scaled_error = torch.inf * torch.tensor((1.0))

    def init(self, func, t0, t1, y0, dt0, args, error_order):
        del t1
        if dt0 is None:
            dt0 = _select_initial_step(
                func, t0, y0, args, error_order, self.rtol, self.atol, rms_norm
            )
            return t0 + dt0, dt0
        else:
            return t0 + dt0, dt0

    def _compute_scaled_error(self, y_error, y0, y1_candidate, norm):
        _nan = torch.any(torch.isnan(y1_candidate))
        y1_candidate = y0 if _nan else y1_candidate
        error_tol = self.atol + self.rtol * torch.max(y0.abs(), y1_candidate.abs())
        return norm(y_error / error_tol).abs().max()

    def update_scaled_error(self, current_scaled_error):
        if not torch.isfinite(self.scaled_error):
            self.scaled_error = current_scaled_error
            return
        if torch.isfinite(self.scaled_error) and not torch.isfinite(
            self.prev_scaled_error
        ):
            self.prev_scaled_error = self.scaled_error
            self.scaled_error = current_scaled_error
            return
        else:
            self.prev_prev_scaled_error = self.prev_scaled_error
            self.prev_scaled_error = self.scaled_error
            self.scaled_error = current_scaled_error
            return

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
        dt,
    ):
        del func, args
        y_error = torch.nan_to_num(y_error, nan=1)
        scaled_error = self._compute_scaled_error(y_error, y0, y1_candidate, rms_norm)
        self.update_scaled_error(scaled_error)
        keep_step = scaled_error <= 1
        factormin = 1.0 if keep_step else 0.1
        scaled_errors = [
            scaled_error,
            self.prev_scaled_error,
            self.prev_prev_scaled_error,
        ]
        new_dt = _optimal_step_size_with_pid(
            dt,
            scaled_errors,
            self.safety,
            self.pcoeff,
            self.icoeff,
            self.dcoeff,
            error_order,
            factormin,
            self.factormax,
        )
        if torch.isinf(new_dt):
            raise ValueError("dt computed is inf")
        if self.dtmin is not None:
            new_dt = torch.max(new_dt, torch.tensor(self.dtmin))
        if self.dtmax is not None:
            new_dt = torch.min(new_dt, torch.tensor(self.dtmax))

        t0 = torch.where(keep_step, t1, t0)
        t1 = torch.where(keep_step, t1 + new_dt, t0 + new_dt)
        return keep_step, t0, t1, new_dt


AdaptiveStepSizeController.__init__.__doc__ = """**Arguments:**

- `atol`: Absolute tolerance.
- `rtol`: Relative tolerance.
- `safety`: Multiplicative safety factor.
- `pcoeff`: The coefficient of the proportional part of the step size control.
- `icoeff`: The coefficient of the integral part of the step size control.
- `dcoeff`: The coefficient of the derivative part of the step size control.
- `factormax`: Maximum amount a step size can be increased relative to the previous
    step.
- `dtmin`: Minimum step size. The step size is either clipped to this value, or an
    error raised if the step size decreases below this, depending on `force_dtmin`.
- `dtmax`: Maximum step size; the step size is clipped to this value.
"""
