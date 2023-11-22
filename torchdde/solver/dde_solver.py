import matplotlib.pyplot as plt
import torch
from torchcubicspline import NaturalCubicSpline, natural_cubic_spline_coeffs

from torchdde.interpolation.linear_interpolation import TorchLinearInterpolator


class DDESolver:
    def __init__(self, solver, delays):
        self.solver = solver
        self.delays = delays

    def __post__init__(self):
        if max(self.delays) <= 0:
            raise "delays must be positive"

    def integrate(self, func, ts, history_func):
        dt = ts[1] - ts[0]
        # y0 should have the shape [batch, N_t=1, features] in order to properly instantiate the
        # interpolator class
        y0 = torch.unsqueeze(history_func(ts[0]).clone(), dim=1)
        ys_interpolation = TorchLinearInterpolator(ts[0].view(1), y0)

        def ode_func(t, y):
            # applies the function func to the current time t and state y and the history
            # we have to make sur that t - tau > dt otherwise we are making a prediction with
            # an unknown ys_interpolation ...
            history = [
                ys_interpolation(t - tau) if t - tau >= ts[0] else history_func(t - tau)
                for tau in self.delays
            ]
            return func(t, y, history=history)

        current_y = y0[:, 0]
        ys = torch.unsqueeze(current_y, dim=1)
        for current_t in ts[:-1]:
            # the stepping method give the next y with a shape [batch, features]
            y = self.solver.step(ode_func, current_t, current_y, dt)
            current_y = y
            # by adding the y to the interpolator, it is unsqueezed in the interpolator class
            ys_interpolation.add_point(current_t + dt, current_y)
            ys = torch.concat((ys, torch.unsqueeze(current_y, dim=1)), dim=1)

        return ys, ys_interpolation

    def integrate_with_cubic_interpolator(self, func, ts, history_func):
        dt = ts[1] - ts[0]
        # y0 should have the shape [batch, N_t=1, features] in order to properly instantiate the
        # TorchLinearInterpolator interpolator class
        y0 = torch.unsqueeze(history_func(ts[0]).clone(), 1)
        ys_interpolation = TorchLinearInterpolator(ts[0].view(1), y0)

        def ode_func(t, y):
            # applies the function func to the current time t and state y and the history
            history = [
                ys_interpolation(t - tau) if t - tau >= ts[0] else history_func(t - tau)
                for tau in self.delays
            ]
            return func(t, y, history=history)

        ys = y0
        current_y = y0[:, 0]
        for i, current_t in enumerate(ts[:-1]):
            # the stepping method give the next y with a shape [batch, features]
            y = self.solver.step(ode_func, current_t, current_y, dt)
            current_y = y
            ys = torch.concat((ys, torch.unsqueeze(y, dim=1)), dim=1)
            # by adding the y to the interpolator, it is unsqueezed in the interpolator class
            # ys_interpolation.add_point(current_t+dt, current_y)
            coeffs = natural_cubic_spline_coeffs(ts[: i + 2], ys)
            ys_interpolation = lambda t: NaturalCubicSpline(coeffs).evaluate(t)

        return ys
