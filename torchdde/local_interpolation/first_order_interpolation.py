import torch


class FirstOrderPolynomialInterpolation:
    # TO FIX FOR GENERAL USE
    # diffrax use evaluate with time scalar
    def __init__(self, t0, t1, dense_info):
        self.t0 = t0
        self.t1 = t1
        self.y0 = dense_info["y0"]
        self.y1 = dense_info["y1"]

    def single_evaluate(self, t):
        dt = self.t1 - self.t0
        dt = torch.where(dt.abs() > 0.0, dt, 1.0)
        coeff = (self.y1 - self.y0) / dt
        print(t)
        return self.y0 + coeff * (t - self.t0)

    def evaluate(self, t):
        return torch.vmap(self.single_evaluate, out_dims=1)(t)
