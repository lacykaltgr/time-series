import torch
import torch.nn as nn

from time_series.recurrent import Recurrent


class VanillaRNN(Recurrent):
    def __init__(self, input_size, hidden_size, **args):
        super(VanillaRNN, self).__init__(
            input_size=input_size,
            units=hidden_size,
            **args
        )
        self.update = nn.Linear(input_size + hidden_size, hidden_size, bias=True)

        self.tanh = nn.Tanh()
        self.init_weights([self.update, self.update_h], "xavier")

    def update_state(self, inputs, state, ts=None):
        combined = torch.cat((inputs, state), dim=1)
        new_state = self.tanh(self.update(combined))
        return new_state, new_state
