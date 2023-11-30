import torch
import torch.nn as nn

from time_series.recurrent import Recurrent


class GRU(Recurrent):
    def __init__(self, input_size, hidden_size: int, **args):
        super(GRU, self).__init__(
            input_size=input_size,
            units=hidden_size,
            **args
        )
        self.update_gate = nn.Linear(input_size + hidden_size, hidden_size, bias=True)
        self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size, bias=True)
        self.new_state = nn.Linear(input_size + hidden_size, hidden_size, bias=True)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.init_weights(
            [self.update_gate, self.reset_gate, self.new_state],
            "xavier"
        )

    def update_state(self, inputs, state, ts=None):
        combined = torch.cat((inputs, state), dim=1)

        update_gate = self.sigmoid(self.update_gate(combined))
        reset_gate = self.sigmoid(self.reset_gate(combined))
        new_memory = self.tanh(
            self.new_state(
                torch.cat((inputs, update_gate * state), dim=1))
        )

        new_state = (1 - reset_gate) * state + reset_gate * new_memory

        return new_state, new_state
