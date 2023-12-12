from typing import List, Optional
import torch.nn as nn
import stribor as st
from torch import Tensor

from time_series.recurrent import Recurrent
from time_series.wiring import FlowWiring


class ResNetFlow(Recurrent):
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
            input_size: int,
            units: List[int],

            n_layers: int,
            time_net: str,
            time_hidden_dim: Optional[int] = None,
            invertible: Optional[bool] = True,
            **kwargs
    ):
        super(ResNetFlow, self).__init__(
            input_size=input_size,
            units=FlowWiring(input_size, n_layers, units, time_hidden_dim),
            **kwargs
        )

        self.resnet_flow = st.net.ResNet(
            input_size,
            units,
            n_layers,
            activation='ReLU',
            final_activation=None,
            time_net=time_net,
            time_hidden_dim=time_hidden_dim,
            invertible=invertible
        )
    def forward(self, inputs, state, ts):
        output = self.resnet_flow(inputs, ts)
        return output, output