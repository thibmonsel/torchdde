import torch


def poly4eval(e, d, c, b, a, t, t0, t1):
    """Evaluate a 4th order polynomial on the interval [t0, t1].

    The coefficients a..e define the polynomial on the interval [0, 1].
    """
    dt = t1 - t0
    dt = torch.where(dt.abs() > 0.0, dt, 1.0)
    x = ((t - t0) / (dt))[:, None].to(dtype=a.dtype)

    # Evaluate the polynomial with Horner's method
    y = a
    y = torch.addcmul(b, y, x)
    y = torch.addcmul(c, y, x)
    y = torch.addcmul(d, y, x)
    y = torch.addcmul(e, y, x)
    return y


class FourthOrderPolynomialInterpolation:
    """Polynomial interpolation on [t0, t1].

    `coefficients` holds the coefficients of a fourth-order polynomial on [0, 1] in
    increasing order, i.e. `cofficients[i]` belongs to `x**i`.
    """

    def __init__(
        self,
        t0,
        t1,
        coefficients,
    ):
        self.t0 = t0
        self.t1 = t1
        self.coefficients = coefficients

    @staticmethod
    def from_k(
        t0,
        t1,
        y0,
        y1,
        k,
        c_mid,
    ):
        # print("k.shape",k.shape)
        dt = (t1 - t0).to(dtype=y0.dtype)
        f0 = dt * k[0]
        f1 = dt * k[-1]
        # print('f0.shape, f1.shape, c_mid.shape',f0.shape, f1.shape, c_mid.shape)
        y_mid = y0 + dt * torch.tensordot(c_mid, k, dims=0)
        # print("y_mid", y_mid.shape)
        a = (2 * (f1 - f0)).add(y1 + y0, alpha=-8).add(y_mid, alpha=16)
        b = (
            (5 * f0)
            .add(f1, alpha=-3)
            .add(y0, alpha=18)
            .add(y1, alpha=14)
            .add(y_mid, alpha=-32)
        )
        c = (
            f1.add(f0, alpha=-4)
            .add(y0, alpha=-11)
            .add(y1, alpha=-5)
            .add(y_mid, alpha=16)
        )
        d = f0
        e = y0
        print(a.shape, d.shape)
        coefficients = (e, d, c, b, a)
        return FourthOrderPolynomialInterpolation(t0, t1, coefficients)

    # def evaluate(self, t):
    #     e, d, c, b, a = self.coefficients
    #     coeff = (e[idx], d[idx], c[idx], b[idx], a[idx])
    #     return poly4eval(*coeff, t, self.t0[idx], self.t1[idx])

    def __repr__(self):
        return (
            f"FourthOrderPolynomialInterpolation(t0={self.t0}, t1={self.t1}, "
            f"coefficients={self.coefficients})"
        )
