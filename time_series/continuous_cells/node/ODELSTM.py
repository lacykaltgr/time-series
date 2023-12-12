# Copyright 2021 The ODE-LSTM Authors. All Rights Reserved.

from time_series.continuous_cells.gated import BaseEncoderLSTM, BaseEncoderGRU
from time_series.continuous_cells.ode import BaseEncoderNODE

#BaseLSTM with BaseEncoderNODE as rnn
class ODELSTM(BaseEncoderLSTM):
    """
        ODE-LSTM model

        Args:
            dim: Data dimension
            hidden_dim: LSTM hidden dimension
            activation: Name of the activation function from `torch.nn`
            final_activation: Name of the activation function from `torch.nn`
            solver: Which numerical solver to use (e.g. `dopri5`, `euler`, `rk4`)
            solver_step: How many solvers steps to take, only applicable for fixed step solvers
    """
    def __init__(
            self,
            input_size: int,
            units: int,

            solver: str,
            solver_step: int,
            **kwargs
    ):
        super(ODELSTM, self).__init__(
            input_size=input_size,
            units=units,
            rnn=BaseEncoderNODE(input_size, units, solver, solver_step),
            jump_first=True,
            **kwargs
        )


#BaseGRU with BaseEncoderNODE as rnn

class ODEGRU(BaseEncoderGRU):
    """
    ODE-LSTM model

    Args:
        dim: Data dimension
        hidden_dim: LSTM hidden dimension
        activation: Name of the activation function from `torch.nn`
        final_activation: Name of the activation function from `torch.nn`
        solver: Which numerical solver to use (e.g. `dopri5`, `euler`, `rk4`)
        solver_step: How many solvers steps to take, only applicable for fixed step solvers
    """
    def __init__(
            self,
            input_size: int,
            units: int,

            solver: str,
            solver_step: int,
            **kwargs
    ):
        super(ODEGRU, self).__init__(
            input_size=input_size,
            units=units,
            rnn=BaseEncoderNODE(input_size, units, solver, solver_step),
            jump_first=True,
            **kwargs
        )





