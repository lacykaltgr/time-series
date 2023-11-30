from torch import nn


from time_series.recurrent import Recurrent
from time_series.cells.discrete.LSTM import LSTM
from time_series.cells.discrete.GRU import GRU


class BaseLSTM(Recurrent):
    """
    Base continuous LSTM class
    Other classes inherit and define `odeint` function

    Args:
        dim: Data dimension
        hidden_dim: Hidden state of LSTM
        odeint: Generic IVP solver, ODE or flow-based model
    """

    def __init__(
            self,
            input_size: int,
            units: int,
            rnn: Recurrent,
            jump_first: bool = True,
            **args
    ):
        super(BaseLSTM, self).__init__(
            input_size=input_size,
            units=units,
            **args
        )
        self.lstm = LSTM(
            input_size=input_size,
            hidden_size=units,
            **args
        )
        self.ctrnn = rnn
        self.jump_first = jump_first

    def update_state(self, input, states, ts=None):
        state, cell_state = states

        if self.jump_first:
            _, lstm_states = self.lstm.update_state(input, states, ts)
            lstm_state, lstm_cell_state = lstm_states

            # Implementation choice on how to parametrize ODE component
            output, state = self.ctrnn.update_state(lstm_state, state, ts)
            # output, state = self.rnn.update_state(lstm_state, lstm_state, ts)
        else:
            output, state = self.ctrnn.update_state(input, state, ts)
            _, lstm_states = self.lstm.update_state(state, states, ts)
            lstm_state, lstm_cell_state = lstm_states
            state = lstm_state

        return output, (state, lstm_cell_state)


class BaseGRU(Recurrent):
    """
    Continuous GRU layer

    Args:
        dim: Data dimension
        hidden_dim: GRU hidden dimension
        model: Which model to use (`ode` or `flow`)
        flow_model: Which flow model to use (currently only `resnet` supported which gives GRU flow)
        flow_layers: How many flow layers
        time_net: Time embedding module
        time_hidden_dim: Time embedding hidden dimension
        solver: Which numerical solver to use
        solver_step: How many solvers steps to take, only applicable for fixed step solvers
    """

    def __init__(
            self,
            input_size: int,
            units: int,
            rnn: Recurrent,
            jump_first: bool = True,
            **args
    ):
        super(BaseGRU).__init__(
            input_size=input_size,
            units=units,
            **args
        )
        self.gru = GRU(
            input_size=input_size,
            hidden_size=units,
            **args
        )
        self.ctrnn = rnn
        self.jump_first = jump_first

    def update_state(self, input, state, ts=None):

        if self.jump_first:
            _, gru_state = self.gru.update_state(input, state, ts)

            # Implementation choice on how to parametrize ODE component
            output, state = self.ctrnn.update_state(gru_state, state, ts)
            # output, state = self.rnn.update_state(gru_state, gru_state, ts)
        else:
            output, state = self.ctrnn.update_state(input, state, ts)
            _, gru_state = self.gru.update_state(state, state, ts)
            state = gru_state

        return output, state
