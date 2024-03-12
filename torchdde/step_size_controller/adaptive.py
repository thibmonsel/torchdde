import torch

from .base import AbstractStepSizeController


class AdaptiveStepSizeController(AbstractStepSizeController):
    def __init__(self, atol, rtol):
        self.atol = atol
        self.rtol = rtol

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
        controller_state,
    ):
        del func, args

        # https://en.wikipedia.org/wiki/Adaptive_step_size
        def _scaled_error(_y0, _y1_candidate, _y_error):
            # In case the solver steps into a region
            # for which the vector field isn't
            # defined.
            _nan = torch.isnan(_y1_candidate).any()
            _y1_candidate = torch.where(_nan, _y0, _y1_candidate)
            _y = torch.max(torch.abs(_y0), torch.abs(_y1_candidate))
            return _y_error / (self.atol + _y * self.rtol)

        scaled_error = _scaled_error(y0, y1_candidate, y_error)  # En
        print("scaled_error", scaled_error)
        keep_step = torch.all(scaled_error < 1)
        print(
            "torch.pow(1/scaled_error, 1/(error_order+1)",
            torch.pow(1 / scaled_error, 1 / (error_order + 1)),
            1 / (error_order),
        )
        next_controller_state = torch.min(controller_state * 1 / scaled_error)
        print("keep_step, t0, t1", keep_step, t0, t1)
        print("controller_state", controller_state)
        print("new_controller_state", next_controller_state)
        next_t0 = t1 if keep_step else t0
        next_t1 = (
            t1 + next_controller_state if keep_step else t0 + next_controller_state
        )
        return keep_step, next_t0, next_t1, next_controller_state

        # error_bounds = torch.add(
        #     self.atol, torch.maximum(y0.abs(), y1_candidate.abs()), alpha=self.rtol
        # )
        # # We lower-bound the error ratio by some small number to avoid division by 0 in
        # # `dt_factor`.
        # error_ratio = self.norm(y_error / error_bounds)
        # keep_step = error_ratio < 1.0

        # # Adapt the step size
        # next_controller_state = torch.min(controller_state * torch.pow(1/error_ratio, 1/(error_order+1)))
        # next_t0 = t1 if keep_step else t0
        # next_t1 = t1 + next_controller_state if keep_step else t0 +next_controller_state
        # return keep_step, next_t0, next_t1, next_controller_state
