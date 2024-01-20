from abc import ABC, abstractmethod

import torch


class AbstractOdeSolver(ABC):
    """Base class for creating ODE solvers. All solvers should inherit from it.
    To create new solvers users must implement the step method.
    """

    @abstractmethod
    def step(
        self,
        func: torch.nn.Module,
        t: torch.Tensor,
        y: torch.Tensor,
        dt: torch.Tensor,
        has_aux=False,
    ) -> torch.Tensor:
        r"""ODE's stepping definition

        **Arguments:**

        - `func`: Pytorch model, i.e vector field
        - `t`: Current time step t
        - `y`: Current state y
        - `dt`: Stepsize dt
        - `has_aux`: Whether the model has an auxiliary output.

        **Returns:**

        Integration result at time `t+dt`
        """
        pass

    def integrate(
        self, func: torch.nn.Module, ts: torch.Tensor, y0: torch.Tensor, has_aux=False
    ) -> torch.Tensor:
        r"""Integrate a system of ODEs.
        **Arguments:**

        - `func`: Pytorch model, i.e vector field
        - `ts`: Integration span
        - `y0`: Initial condition
        - `has_aux`: Whether the model has an auxiliary output.

        **Returns:**

        Integration result over `ts`
        """

        dt = ts[1] - ts[0]
        ys = torch.unsqueeze(y0.clone(), dim=1)
        current_y = y0
        for current_t in ts[1:]:
            y = self.step(func, current_t, current_y, dt, has_aux=has_aux)
            current_y = y
            ys = torch.cat((ys, torch.unsqueeze(current_y, dim=1)), dim=1)
        return ys


class Euler(AbstractOdeSolver):
    """Euler's method"""

    def __init__(self):
        super().__init__()

    def step(self, func, t, y, dt, has_aux=False):
        if has_aux:
            k1, aux = func(t, y, has_aux)
            return y + dt * k1, aux
        else:
            return y + dt * func(t, y)


class RK2(AbstractOdeSolver):
    """2nd order explicit Runge-Kutta method"""

    def __init__(self):
        super().__init__()

    def step(self, func, t, y, dt, has_aux=False):
        if has_aux:
            k1, aux = func(t, y, has_aux)
            k2 = func(t + dt, y + dt * k1)
            return y + dt / 2 * (k1 + k2), aux
        else:
            k1 = func(t, y)
            k2 = func(t + dt, y + dt * k1)
            return y + dt / 2 * (k1 + k2)


class Ralston(AbstractOdeSolver):
    """Ralston's method (2nd order)"""

    def __init__(self):
        super().__init__()

    def step(self, func, t, y, dt, has_aux=False):
        if has_aux:
            k1, aux = func(t, y, has_aux)
            k2 = func(t + 2 / 3 * dt, y + 2 / 3 * dt * k1)
            return y + dt * (1 / 4 * k1 + 3 / 4 * k2), aux
        else:
            k1 = func(t, y)
            k2 = func(t + 2 / 3 * dt, y + 2 / 3 * dt * k1)
            return y + dt * (1 / 4 * k1 + 3 / 4 * k2)


class RK4(AbstractOdeSolver):
    """4th order explicit Runge-Kutta method"""

    def __init__(self):
        super().__init__()

    def step(self, func, t, y, dt, has_aux=False):
        if has_aux:
            k1, aux = func(t, y, has_aux)
            k2 = func(t + 1 / 2 * dt, y + 1 / 2 * dt * k1)
            k3 = func(t + 1 / 2 * dt, y + 1 / 2 * dt * k2)
            k4 = func(t + dt, y + dt * k3)
            return y + 1 / 6 * dt * (k1 + 2 * k2 + 2 * k3 + k4), aux
        else:
            k1 = func(t, y)
            k2 = func(t + 1 / 2 * dt, y + 1 / 2 * dt * k1)
            k3 = func(t + 1 / 2 * dt, y + 1 / 2 * dt * k2)
            k4 = func(t + dt, y + dt * k3)
            return y + 1 / 6 * dt * (k1 + 2 * k2 + 2 * k3 + k4)
