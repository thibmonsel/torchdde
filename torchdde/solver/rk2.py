from torchdde.local_interpolation import FirstOrderPolynomialInterpolation
from torchdde.solver.runge_kutta import ButcherTableau, ExplicitRungeKutta


class RK2(ExplicitRungeKutta):
    """
    Second-order explicit Runge-Kutta method (RK2's method).

    This implementation uses the 2-stage Butcher Tableau for Heun's method
    (also known as the improved Euler method or trapezoidal rule).
    It does not inherently provide error estimation for adaptive step sizing.
    """

    def __init__(self):
        c_list = [0.0, 1.0]
        a_list = [[], [1.0]]
        b_list = [1 / 2, 1 / 2]
        b_err_list = [0.0, 0.0]

        # Create the ButcherTableau object
        rk2_tableau = ButcherTableau.from_lists(
            c=c_list, a=a_list, b=b_list, b_err=b_err_list
        )

        super().__init__(
            tableau=rk2_tableau,
            interpolation_cls=FirstOrderPolynomialInterpolation,
        )

    def order(self) -> int:
        return 2
