import functools
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import torch
from interpolators import TorchLinearInterpolator
from ode_solver import AbstractOdeSolver, Euler, Ralston, RK4
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline


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
            history = [
                ys_interpolation(t - tau) if t - tau >= ts[0] else history_func(t - tau)
                for tau in self.delays
            ]
            # print("ode func", y.shape, history[0].shape)
            return func(t, y, history=history)

        current_y = y0[:, 0]
        ys = torch.unsqueeze(current_y, dim=1)
        for current_t in ts[:-1]:
            # the stepping method give the next y with a shape [batch, features]
            # print("current y", current_t, current_y.shape)
            y = self.solver.step(ode_func, current_t, current_y, dt)
            current_y = y
            # by adding the y to the interpolator, it is unsqueezed in the interpolator class
            ys_interpolation.add_point(current_t + dt, current_y)
            ys = torch.concat((ys, torch.unsqueeze(current_y, dim=1)), dim=1)

        return ys, ys_interpolation

    def integrate_with_cubic_interpolator(self, func, ts, history_func):
        dt = ts[1] - ts[0]
        # y0 should have the shape [batch, N_t=1, features] in order to properly instantiate the
        # interpolator class
        y0 = torch.unsqueeze(history_func(ts[0]).clone(), 1)
        ys_interpolation = TorchLinearInterpolator(ts[0].view(1), y0)
        # coeffs = natural_cubic_spline_coeffs(ts[0].view(1), y0)
        # ys_interpolation = NaturalCubicSpline(coeffs)

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


# def simple_dde(t, y, *, history):
#     return -history[0]

# def simple_dde2(t, y, *, history):
#     return 0.25 * (history[0]) / (1.0 + history[0] ** 10) - 0.1 * y
#     # return y * (1.0 - history[0])

# history_function = lambda t: torch.tensor([[0.0, 1.0, 2.0, 3.0]])
# ts = torch.linspace(0, 100, 401)
# solver = Ralston()
# dde_solver = DDESolver(solver, [10.0])
# ys = dde_solver.integrate(simple_dde, ts, history_function)
# # print(ys[0])

# import time


# t = time.time()
# ys2 = dde_solver.integrate(simple_dde2, ts, history_function)
# print(time.time()- t)

# t = time.time()
# ysbis = dde_solver.integrate_with_cubic_interpolator(simple_dde2, ts, history_function)
# print(time.time() - t )

# plt.plot(ts, ys2[0], '--')
# plt.plot(ts, ysbis[0])
# plt.show()
