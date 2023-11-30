

class LSTMResNet(BaseContinuousLSTM):
    """
    LSTM-based ResNet flow

    Args:
        dim: Data dimension
        hidden_dim: LSTM hidden dimension
        n_layers: How many flow layers
        time_net: Time embedding module
        time_hidden_dim: Time embedding hidden dimension
    """
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            n_layers: int,
            time_net: str,
            time_hidden_dim: Optional[int] = None,
            **kwargs
    ):
        super().__init__(
            dim,
            hidden_dim,
            ResNetFlow(
                dim=dim,
                n_layers=n_layers,
                hidden_dims=[hidden_dim],
                time_net=time_net,
                time_hidden_dim=time_hidden_dim
            )
        )


class LSTMCoupling(BaseContinuousLSTM):
    """
    LSTM-based coupling flow

    Args:
        dim: Data dimension
        hidden_dim: LSTM hidden dimension
        n_layers: How many flow layers
        time_net: Time embedding module
        time_hidden_dim: Time embedding hidden dimension
    """
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            n_layers: int,
            time_net: str,
            time_hidden_dim: Optional[int] = None,
            **kwargs
    ):
        super().__init__(
            dim,
            hidden_dim,
            CouplingFlow(
                dim=hidden_dim,
                n_layers=n_layers,
                hidden_dims=[hidden_dim],
                time_net=time_net,
                time_hidden_dim=time_hidden_dim
            )
        )