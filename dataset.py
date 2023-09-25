import numpy as np
import torch

from torchdde.interpolation.linear_interpolation import TorchLinearInterpolator


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
