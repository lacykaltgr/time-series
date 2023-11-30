
from typing import List, Optional
import torch.nn as nn
import stribor as st
from torch import Tensor

from time_series.recurrent import Recurrent
from time_series.wiring import FlowWiring


class CouplingFlow(Recurrent):
    """
    Affine coupling flow

    Args:
        dim: Data dimension
        n_layers: Number of flow layers
        hidden_dims: Hidden dimensions of the flow neural network
        time_net: Time embedding module
        time_hidden_dim: Time embedding hidden dimension
    """
    def __init__(
            self,
            input_size: int,
            n_layers: int,
            hidden_dims: List[int],
            time_net: str,
            time_hidden_dim: Optional[int] = None,
            **kwargs
    ):
        super().__init__(
            input_size=input_size,
            units=FlowWiring(input_size,n_layers, hidden_dims, time_hidden_dim),
            **kwargs
        )

        time_nets = [
            "TimeLog", "TimeTanh", "TimeFourier", "TimeFourierBounded", "TimeIdentity", "TimeLinear"
        ]

        transforms = []
        dim = input_size
        for i in range(n_layers):
            transforms.append(st.ContinuousAffineCoupling(
                latent_net=st.net.MLP(dim + 1, hidden_dims, 2 * dim),
                time_net=getattr(st.net.Tim, time_net)(2 * dim, hidden_dim=time_hidden_dim),
                mask='none' if dim == 1 else f'ordered_{i % 2}'))
        self.flow = st.NeuralFlow(transforms=transforms)

    def update_state(self, inputs, state, ts):

        if inputs.shape[-2] == 1:
            inputs = inputs.repeat_interleave(ts.shape[-2], dim=-2) # (..., 1, dim) -> (..., seq_len, 1)

        # If t0 not 0, solve inverse first
        if t0 is not None:
            x = self.flow.inverse(x, t=t0)[0]

        self.flow()

        return self.flow(x, t=t)[0]

