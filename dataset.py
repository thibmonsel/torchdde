import numpy as np
import torch
from scipy.integrate import solve_ivp
from torch.utils.data import Dataset

from torchdde.interpolation.linear_interpolation import TorchLinearInterpolator


class MyDataset(Dataset):
    def __init__(self, ys):
        self.ys = ys

    def __getitem__(self, index):
        return self.ys[index]

    def __len__(self):
        return self.ys.shape[0]

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


def ks(dataset_size, ts, L=22, N=128, nu=1):
    k = 2 * np.pi * np.fft.rfftfreq(N, d=L / N)
    u0_val = np.zeros((dataset_size, N // 2 + 1,))
    u0_val[:, :4] = np.random.normal(size=(dataset_size, 4,))
    u0_val = np.fft.irfft(u0_val, axis=-1)

    def vector_field(t, u):
        u_hat = np.fft.rfft(u)
        rhs_freq = (
            k**2 - nu * k**4
        ) * u_hat - 1 / 2 * 1j * k * np.fft.rfft(u**2)
        return np.fft.irfft(rhs_freq)
    
    sol = np.empty((dataset_size, len(ts), u0_val.shape[-1]))
    for i, y0_ in enumerate(u0_val):
        sol[i] = solve_ivp(vector_field, (ts[0], ts[-1]), y0_, t_eval=ts, method="Radau").y.T
    return torch.from_numpy(sol) 

def lorenz(y0, ts, args):
    sigma, rho, beta = args
    def vector_field(t, x):
        x_rhs = sigma * (x[1] - x[0])
        y_rhs = x[0] * (rho - x[2]) - x[1]
        z_rhs = x[0]*x[1] - beta * x[2]
        return np.stack([x_rhs, y_rhs, z_rhs], axis=-1)

    sol = np.empty((len(y0), len(ts), 1))
    for i, y0_ in enumerate(y0):
        sol[i] = solve_ivp(vector_field, (ts[0], ts[-1]), y0_, t_eval=ts).y[0][..., None]
    return torch.from_numpy(sol) 





def burgers(dataset_size, ts, xs):
    mu = 1
    nu = 8 * 10 ** (-4)  # kinematic viscosity coefficient
    # k0 = 10
    # A = 2 * k0**(-5) / (3 * np.sqrt(jnp.pi))
    k = 2 * np.pi * np.fft.fftfreq(xs.shape[0], d=(xs[1] - xs[0]))
    # E0 = A * k**4 * np.exp(-(k/k0)**2)
    # u0 = np.sqrt(2 * E0)  * 
    # ( np.cos( 2 *np.pi * psi_k)- np.sin( 2 *np.pi * psi_k))
    u0 = np.zeros((dataset_size, xs.shape[0] // 2 + 1,))
    u0[:, :20] = 10 * np.random.normal(size=(dataset_size, 20,))
    u0 = np.fft.irfft(u0, axis=-1)
    # Def of the initial condition
    
    def vf_burgers(t, u):
        # Definition of ODE system (PDE ---(FFT)---> ODE system)
        # Spatial derivative in the Fourier domain
        u_hat = np.fft.fft(u)
        u_hat_x = 1j * k * u_hat
        u_hat_xx = -(k**2) * u_hat

        # Switching in the spatial domain
        u_x = np.fft.ifft(u_hat_x)
        u_xx = np.fft.ifft(u_hat_xx)

        # ODE resolution
        u_t = -mu * u * u_x + nu * u_xx
        return u_t.real
    
    sol = np.empty((dataset_size, len(ts), u0.shape[-1]))

    for i, y0_ in enumerate(u0):
        sol[i] = solve_ivp(vf_burgers, (ts[0], ts[-1]), y0_, t_eval=ts, method="Radau").y.T
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
