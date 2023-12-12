from typing import List, Optional
import stribor as st

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
            units: List[int],

            n_layers: int,
            time_net: str,
            time_hidden_dim: Optional[int] = None,
            **kwargs
    ):
        super().__init__(
            input_size=input_size,
            units=FlowWiring(input_size, n_layers, units, time_hidden_dim),
            **kwargs
        )

        time_nets = [
            "TimeLog", "TimeTanh", "TimeFourier", "TimeFourierBounded", "TimeIdentity", "TimeLinear"
        ]

        transforms = []
        for i in range(n_layers):
            transforms.append(st.ContinuousAffineCoupling(
                latent_net=st.net.MLP(input_size + 1, units, 2 * input_size),
                time_net=getattr(st.net, "time_net")(2 * input_size, hidden_dim=time_hidden_dim),
                mask='none' if input_size == 1 else f'ordered_{i % 2}'))
        self.flow = st.NeuralFlow(transforms=transforms)

    def update_state(self, inputs, state, ts):
        return self.flow(inputs, t=ts)

