from time_series.continuous_cells.gated import BaseLSTM
from typing import Optional, Union, List
from time_series.continuous_cells.flows.ResNetFlow import ResNetFlow
from time_series.continuous_cells.flows.CouplingFlow import CouplingFlow
from time_series.continuous_cells.flows.GRUFlow import GRUFlow


class FlowLSTM(BaseLSTM):
    """
    LSTM-based ResNet/Coupling/GRU flow

    Args:
        dim: Data dimension
        hidden_dim: LSTM hidden dimension
        n_layers: How many flow layers
        time_net: Time embedding module
        time_hidden_dim: Time embedding hidden dimension
    """
    def __init__(
            self,
            input_size: int,
            units: int,

            flow_model: str,
            n_layers: int,
            time_net: str,
            time_hidden_dim: Optional[int] = None,
            **kwargs
    ):
        if flow_model == "resnet":
            flow = ResNetFlow
        elif flow_model == "coupling":
            flow = CouplingFlow
        elif flow_model == "gru":
            flow = GRUFlow
        else:
            raise ValueError(f"Unknown flow model {flow_model}")

        flow = flow(
                input_size=input_size,
                units=[units],

                n_layers=n_layers,
                time_net=time_net,
                time_hidden_dim=time_hidden_dim
            )

        super(FlowLSTM, self).__init__(
            input_size,
            units,
            rnn=flow,
            **kwargs
        )