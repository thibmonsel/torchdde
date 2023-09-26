import numpy as np
import torch
from scipy.integrate import solve_ivp

from torchdde.interpolation.linear_interpolation import TorchLinearInterpolator


def stiff_vdp(y0, ts):
    """Generate a VdP dataset :
    d2x/dt2 - eps * w0 (1 - x^2)dx/dt + w0^2*x = 0

    Args:
        nb_trajectories (int): number od datapoints
        eps (float, optional): parameters of VdP equation. Defaults to 1.
        w0 (float, optional): parameters of VdP equation. Defaults to 1.

    http://www.cax.free.fr/vdp/vdp.html
    """

    def vector_field(t, x):
        x_rhs = x[1]
        xdot_rhs = 1.5 * (1 - x[0] ** 2) * x[1] - x[0]
        return np.stack([x_rhs, xdot_rhs], axis=-1)

    sol = np.empty((len(y0), len(ts), 1))
    for i, y0_ in enumerate(y0):
        sol[i] = solve_ivp(vector_field, (ts[0], ts[-1]), y0_, t_eval=ts).y[1][..., None]
    return torch.from_numpy(sol) 


def brusellator(y0, ts, args):
    a, b = args
    def vector_field(t, x):
        x1, x2 = x
        dxdt = a - (1 + b) * x1 + x1**2 * x2
        dydt = b * x1 - x1**2 * x2
        return np.stack(
            [
                dxdt,
                dydt,
            ],
            axis=-1,
        )

    sol = np.empty((len(y0), len(ts), 1))
    for i, y0_ in enumerate(y0):
        sol[i] = solve_ivp(vector_field, (ts[0], ts[-1]), y0_, t_eval=ts).y[0][..., None]
    return torch.from_numpy(sol) 



def get_batch(
    ts,
    ys,
    list_delays,
    device="cpu",  # torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """
    ts : [N_t]
    ys : [B, N_t, #features]
    """
    dt = ts[1] - ts[0]
    max_delay = max(list_delays)
    max_delay_idx = int(max_delay / dt)
    # pick random indices for each batch
    rand_idx = np.random.choice(ys.shape[1] - max_delay_idx - length - 1)
    # history_batch : [batch_size, max_delay_idx, #features]
    # ts_history : [length] negative time
    # data_batch : [batch_size, length, #features]
    history_batch = ys[:, rand_idx : rand_idx + max_delay_idx + 1]
    ts_history = torch.linspace(0, max_delay, max_delay_idx + 1)
    data_batch = ys[:, rand_idx + max_delay_idx : rand_idx + max_delay_idx + length + 1]
    ts_data = torch.linspace(max_delay, float(length * dt), length + 1)
    interpolator = TorchLinearInterpolator(ts_history, history_batch, device)
    data_batch = data_batch.to(device)
    return interpolator, ts_data, data_batch
