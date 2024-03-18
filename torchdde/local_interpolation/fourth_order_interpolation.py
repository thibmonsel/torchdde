import torch


def linear_rescale(t0, t, t1):
    """
    Calculates (t - t0) / (t1 - t0), assuming t0 <= t <= t1.
    """

    cond = t0 == t1
    numerator = torch.zeros_like(t) if cond else t - t0
    denominator = torch.ones_like(t) if cond else t1 - t0
    return numerator / denominator


class FourthOrderPolynomialInterpolation:
    """Polynomial interpolation on [t0, t1].

    `coefficients` holds the coefficients of a fourth-order polynomial on [0, 1] in
    increasing order, i.e. `cofficients[i]` belongs to `x**i`.
    """

    def __init__(self, t0, t1, y0, y1, k, c_mid):
        self.t0 = t0
        self.t1 = t1
        self.coeffs = self._calculate(y0, y1, k, c_mid)

    def _calculate(self, _y0, _y1, _k, c_mid):
        dt = self.t1 - self.t0
        _ymid = _y0 + dt * torch.einsum("c, cbf -> bf", c_mid, _k)
        _f0 = dt * _k[0]
        _f1 = dt * _k[-1]

        _a = 2 * (_f1 - _f0) - 8 * (_y1 + _y0) + 16 * _ymid
        _b = 5 * _f0 - 3 * _f1 + 18 * _y0 + 14 * _y1 - 32 * _ymid
        _c = _f1 - 4 * _f0 - 11 * _y0 - 5 * _y1 + 16 * _ymid
        return torch.stack([_a, _b, _c, _f0, _y0], dim=1).type(torch.float32)

    def evaluate(self, t, t1=None, left: bool = True):
        del left
        if t1 is not None:
            return self.evaluate(t1) - self.evaluate(t)

        t = linear_rescale(self.t0, t, self.t1)
        t_polynomial = torch.pow(
            t * torch.ones((5,)),
            exponent=torch.flip(torch.arange(5), dims=(0,)),  # pyright : ignore
        )
        # print("t_po", t_polynomial.shape, self.coeffs.shape)
        return torch.einsum("c, bcf -> bf", t_polynomial, self.coeffs)
        # return torch.einsum("cp, cbf -> bpf", t_polynomial, self.coeffs)

    def __repr__(self):
        return (
            f"FourthOrderPolynomialInterpolation(t0={self.t0}, t1={self.t1}, "
            f"coefficients={self.coeffs})"
        )
