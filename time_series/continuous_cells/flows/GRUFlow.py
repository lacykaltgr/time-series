import torch
from torch import nn
import stribor as st

from time_series.recurrent import Recurrent

class GRUFlow(Recurrent):
    """
    Single GRU flow layer

    Args:
        hidden_dim: Size of the GRU hidden state
        time_net: Time embedding module
        time_hidden_dim: Time embedding hidden dimension
    """
    def __init__(
            self,
            hidden_dim,
            time_net,
            time_hidden_dim=None
    ):
        super().__init__(input_size=hidden_dim, units=hidden_dim)

        # Spectral norm for linear layers
        norm = lambda layer: torch.nn.utils.spectral_norm(layer, n_power_iterations=5)

        self.lin_hh = norm(nn.Linear(hidden_dim + 1, hidden_dim))
        self.lin_hz = norm(nn.Linear(hidden_dim + 1, hidden_dim))
        self.lin_hr = norm(nn.Linear(hidden_dim + 1, hidden_dim))

        self.time_net = getattr(st.net, time_net)(hidden_dim, hidden_dim=time_hidden_dim)

        # Additional constants that ensure invertibility, see Theorem 1 in paper
        self.alpha = 2 / 5
        self.beta = 4 / 5

    def residual(self, h, t):
        inp = torch.cat([h, t], -1)
        r = self.beta * torch.sigmoid(self.lin_hr(inp))
        z = self.alpha * torch.sigmoid(self.lin_hz(inp))
        u = torch.tanh(self.lin_hh(torch.cat([r * h, t], -1)))
        return z * (u - h)

    def update_state(self, inputs, state, t):
        state = state + self.time_net(t) * self.residual(state, t)
        return state

    def inverse(self, y, t, iterations=100):
        x = y
        for _ in range(iterations):
            residual = self.time_net(t) * self.residual(x, t)
            x = y - residual
        return x

"""
    def inverse(self, y, t):
        for layer in reversed(self.layers):
            y = layer.inverse(y, t)
        return y
"""