import torch
import torch.nn as nn


class NDDE(nn.Module):
    def __init__(self, dim, delays, width=64):
        super().__init__()
        self.in_dim = dim * (1 + len(delays))
        self.delays = torch.nn.Parameter(delays)
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, dim),
        )

    def forward(self, t, z, *, history):
        
        inp = torch.cat([z, *history], dim=-1)
        return self.mlp(inp)


class ConvNDDE(nn.Module):
    def __init__(self, dim, delays):
        super().__init__()
        self.in_dim = dim * (1 + len(delays))
        self.delays = torch.nn.Parameter(delays)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=self.in_dim, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=1, kernel_size=3, padding=1),
        )
        self.net2 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=1, kernel_size=3, padding=1),
        )

    def forward(self, t, z, *, history):
        inp = torch.cat([torch.unsqueeze(z, dim=1), *[torch.unsqueeze(h, dim=1) for h in history]], dim=1)
        return self.net(inp)[:, 0] +  self.net2(torch.unsqueeze(z, dim=1))[:, 0]



class SimpleNDDE(nn.Module):
    def __init__(self, dim, list_delays):
        super().__init__()
        self.in_dim = dim * (1 + len(list_delays))
        self.delays = list_delays
        self.linear = torch.nn.Linear(self.in_dim, 1, bias=False)

    def init_weight(self, value):
        with torch.no_grad():
            self.linear.weight = nn.Parameter(torch.tensor([[value, -value]]))

    def forward(self, t, z, *, history):
        z__history = z * history[0]
        inp = torch.cat([z, z__history], dim=-1)
        return self.linear(inp)


class SimpleNDDE2(nn.Module):
    def __init__(self, dim, list_delays):
        super().__init__()
        self.in_dim = dim * (1 + len(list_delays))
        self.delays = list_delays
        self.linear = torch.nn.Linear(self.in_dim, 1, bias=False)

    def init_weight(self, value):
        with torch.no_grad():
            self.linear.weight = nn.Parameter(torch.tensor([[value, -value]]))

    def forward(self, t, z, *, history):
        # with init_weight this is equivalent to theta_0 * z * (1 - theta_1 * history)
        inp = torch.cat([z, *history], dim=-1)
        return self.linear(inp)
