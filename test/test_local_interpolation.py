import torch
from torchdde import ThirdOrderPolynomialInterpolation


def test_third_order_interpolation():
    t0 = torch.tensor(2.0)
    t1 = torch.tensor(3.9)

    def y(t):
        return 0.4 + 0.7 * t - 1.1 * t**2 + 0.4 * t**3

    def yprime(t):
        return 0.7 - 2.2 * t + 1.2 * t**2

    y0, f0 = y(t0), yprime(t0)
    y1, f1 = y(t1), yprime(t1)
    k0 = f0 * (t1 - t0)
    k1 = f1 * (t1 - t0)
    y0, f0, y1, f1, k0, k1 = (
        y0.reshape(-1, 1),
        f0.reshape(-1, 1),
        y1.reshape(-1, 1),
        f1.reshape(-1, 1),
        k0.reshape(-1, 1),
        k1.reshape(-1, 1),
    )
    dense_info = dict(y0=y0, y1=y1, k=torch.stack([k0, k1]))
    interp = ThirdOrderPolynomialInterpolation(t0=t0, t1=t1, dense_info=dense_info)
    assert torch.allclose(
        interp(torch.tensor(2.6)).flatten(),
        interp(torch.tensor(2.6)).flatten(),
    )
