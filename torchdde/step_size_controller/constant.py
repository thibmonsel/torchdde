import torch

from .base import AbstractStepSizeController


class ConstantStepSizeController(AbstractStepSizeController):
    """Constant step size controller that always returns the same step size.

    The user must define `dt0` via [`torchdde.integrate.integrate`][]"""

    def init(self, func, t0, t1, y0, dt0, args, error_order):
        del func, t1, y0, args, error_order
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
        dt,
    ):
        del func, t0, y0, y1_candidate, args, y_error, error_order
        return torch.tensor([True], device=t1.device), t1, t1 + dt, dt
