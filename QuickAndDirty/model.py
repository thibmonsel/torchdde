import torch
import torch.nn as nn


class NDDE(nn.Module):
    def __init__(self, dim, list_delays, width=64):
        super().__init__()
        self.in_dim = dim * (1 + len(list_delays))
        self.delays = list_delays
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, dim),
        )

    def forward(self, t, z, *, history):
        inp = torch.cat([z, *history], dim=-1)
        return self.mlp(inp)


class SimpleNDDE(nn.Module):
    def __init__(self, dim, list_delays):
        super().__init__()
        self.in_dim = dim * (1 + len(list_delays))
        self.delays = list_delays
        self.linear = torch.nn.Linear(self.in_dim, 1, bias=False)

    def init_weight(self, value):
        with torch.no_grad():
            self.linear.weight = nn.Parameter(torch.tensor([value, value]))

    def forward(self, t, z, *, history):
        # print('z, hist', z.shape, history[0].shape)
        inp = torch.cat([z, *history], dim=-1)
        # print('self.linear(inp)', self.linear(inp).shape, inp.shape)
        return self.linear(inp)


# class SimpleNDDE(nn.Module):
#     def __init__(self, dim, list_delays):
#         super().__init__()
#         self.in_dim = dim * (1 + len(list_delays))
#         self.delays = list_delays
#         self.linear = torch.nn.Linear(self.in_dim, 1, bias=False)

#     def init_weight(self, value):
#         with torch.no_grad():
#             self.linear.weight = nn.Parameter(torch.tensor([value, -value]))

#     def forward(self, t, z, *, history):
#         # with init_weight this is equivalent to theta_0 * z * (1 - theta_1 * history)
#         z__history = z * history[0]
#         inp = torch.cat([z, z__history], dim=-1)
#         return self.linear(inp)
