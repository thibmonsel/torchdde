from torchdde.local_interpolation import ThirdOrderPolynomialInterpolation
from torchdde.solver.runge_kutta import ButcherTableau, ExplicitRungeKutta


class Bosh3(ExplicitRungeKutta):
    """
    Bogacki--Shampine's 3/2 method.

    3rd order explicit Runge--Kutta method. Has an embedded 2nd order method for
    adaptive step sizing. Uses 4 stages with FSAL. Uses 3rd order Hermite
    interpolation for dense/ts output.
    """

    c_list = [0.0, 1 / 2, 3 / 4, 1.0]
    a_list = [
        [],
        [1 / 2],
        [0.0, 3 / 4],
        [2 / 9, 1 / 3, 4 / 9],
    ]
    b_list = [2 / 9, 1 / 3, 4 / 9, 0.0]
    b_low_order = [7 / 24, 1 / 4, 1 / 3, 1 / 8]

    def __init__(self):
        bosh3_tableau = ButcherTableau.from_lists(
            c=self.c_list,
            a=self.a_list,
            b=self.b_list,
            b_low_order=self.b_low_order,
        )

        super().__init__(
            tableau=bosh3_tableau,
            interpolation_cls=ThirdOrderPolynomialInterpolation,
        )

    def order(self) -> int:
        return 3
