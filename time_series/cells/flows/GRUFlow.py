
class GRUFlowBlock(Module):
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
        super().__init__()

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

    def forward(self, h, t):
        h = h + self.time_net(t) * self.residual(h, t)
        return h

    def inverse(self, y, t, iterations=100):
        x = y
        for _ in range(iterations):
            residual = self.time_net(t) * self.residual(x, t)
            x = y - residual
        return x


class GRUFlow(Module):
    """
    GRU flow model

    Args:
        dim: Data dimension
        n_layers: Number of flow layers
        time_net: Time embedding module
        time_hidden_dim: Time embedding hidden dimension
    """
    def __init__(
            self,
            dim: int,
            n_layers: int,
            time_net: str,
            time_hidden_dim: Optional[int] = None,
            **kwargs
    ):
        super().__init__()

        layers = []
        for _ in range(n_layers):
            layers.append(GRUFlowBlock(dim, time_net, time_hidden_dim))

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        if x.shape[-2] != t.shape[-2]:
            x = x.repeat_interleave(t.shape[-2], dim=-2)

        for layer in self.layers:
            x = layer(x, t)

        return x

    def inverse(self, y, t):
        for layer in reversed(self.layers):
            y = layer.inverse(y, t)
        return y