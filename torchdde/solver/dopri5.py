import torch

from torchdde.local_interpolation.fourth_order_interpolation import (
    FourthOrderPolynomialInterpolation,
)
from torchdde.solver.runge_kutta import ButcherTableau, ExplicitRungeKutta


class _Dopri5Interpolation(FourthOrderPolynomialInterpolation):
    c_mid = torch.tensor(
        [
            6025192743 / 30085553152 / 2,
            0,
            51252292925 / 65400821598 / 2,
            -2691868925 / 45128329728 / 2,
            187940372067 / 1594534317056 / 2,
            -1776094331 / 19743644256 / 2,
            11237099 / 235043384 / 2,
        ],
    )

    def __init__(self, t0, t1, dense_info):
        super().__init__(t0, t1, dense_info, self.c_mid)

    def init(self):
        pass


class Dopri5(ExplicitRungeKutta):
    """
    Dormand-Prince 5(4) explicit Runge-Kutta method.

    Uses a 7-stage, 5th-order method with an embedded 4th-order method for
    error estimation and adaptive step sizing. Features the FSAL property.
    """

    c_list = [0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0]
    a_list = [
        [],
        [1 / 5],
        [3 / 40, 9 / 40],
        [44 / 45, -56 / 15, 32 / 9],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
        [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
    ]
    b_list = [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0]
    b_err_list = [
        (35 / 384 - 5179 / 57600),
        (0.0 - 0.0),
        (500 / 1113 - 7571 / 16695),
        (125 / 192 - 393 / 640),
        (-2187 / 6784 - -92097 / 339200),
        (11 / 84 - 187 / 2100),
        (0.0 - 1 / 40),
    ]

    def __init__(self):
        dopri5_tableau = ButcherTableau.from_lists(
            c=self.c_list, a=self.a_list, b=self.b_list, b_err=self.b_err_list
        )

        super().__init__(
            tableau=dopri5_tableau,
            interpolation_cls=_Dopri5Interpolation,
        )

    def order(self):
        return 5
