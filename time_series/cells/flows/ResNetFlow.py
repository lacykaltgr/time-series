from typing import List, Optional
import torch.nn as nn
import stribor as st
from torch import Tensor

from time_series.recurrent import Recurrent
from time_series.wiring import FlowWiring


class ResNetFlow(Module):
    """
    ResNet flow

    Args:
        dim: Data dimension
        n_layers: Number of flow layers
        hidden_dims: Hidden dimensions of the residual neural network
        time_net: Time embedding module
        time_hidden_dim: Time embedding hidden dimension
        invertible: Whether to make ResNet invertible (necessary for proper flow)
    """
    def __init__(
            self,
            dim: int,
            n_layers: int,
            hidden_dims: List[int],
            time_net: str,
            time_hidden_dim: Optional[int] = None,
            invertible: Optional[bool] = True,
            **kwargs
    ):
        super().__init__()

        layers = []
        for _ in range(n_layers):
            layers.append(st.net.ResNetFlow(
                dim,
                hidden_dims,
                n_layers,
                activation='ReLU',
                final_activation=None,
                time_net=time_net,
                time_hidden_dim=time_hidden_dim,
                invertible=invertible
            ))

        self.layers = nn.ModuleList(layers)

    def forward(
            self,
            x: Tensor, # Initial conditions, (..., 1, dim)
            t: Tensor, # Times to solve at, (..., seq_len, dim)
    ) -> Tensor: # Solutions to IVP given x at t, (..., times, dim)

        if x.shape[-2] == 1:
            x = x.repeat_interleave(t.shape[-2], dim=-2)

        for layer in self.layers:
            x = layer(x, t)

        return x