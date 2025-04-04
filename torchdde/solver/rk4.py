from torchdde.local_interpolation import ThirdOrderPolynomialInterpolation
from torchdde.solver.runge_kutta import ButcherTableau, ExplicitRungeKutta


# Define the RK4 class inheriting from ExplicitRungeKutta
class RK4(ExplicitRungeKutta):
    """
    Classic fourth-order explicit Runge-Kutta method.

    This implementation uses the standard Butcher Tableau for RK4.
    It does not inherently provide error estimation for adaptive step sizing.
    """

    def __init__(self):
        c_list = [0.0, 1 / 2, 1 / 2, 1.0]

        a_list = [[], [1 / 2], [0.0, 1 / 2], [0.0, 0.0, 1.0]]

        b_list = [1 / 6, 1 / 3, 1 / 3, 1 / 6]
        # Standard RK4 doesn't have an embedded method for error estimation
        # by default, so b_err is [0, 0, 0, 0].
        b_err_list = [0.0, 0.0, 0.0, 0.0]

        # Create the ButcherTableau object
        rk4_tableau = ButcherTableau.from_lists(
            c=c_list, a=a_list, b=b_list, b_err=b_err_list
        )

        super().__init__(
            tableau=rk4_tableau,
            interpolation_cls=ThirdOrderPolynomialInterpolation,
        )

    def order(self) -> int:
        return 4
