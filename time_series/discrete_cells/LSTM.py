# Copyright 2021 The ODE-LSTM Authors. All Rights Reserved.

import torch
from torch import nn

from time_series.recurrent import Recurrent


class LSTM(Recurrent):

    def __init__(self, input_size, hidden_size: int, **args):
        super(LSTM, self).__init__(
            input_size=input_size,
            units=hidden_size,
            **args
        )
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size, bias=True)
        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size, bias=True)
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size, bias=True)
        self.cell_gate = nn.Linear(input_size + hidden_size, hidden_size, bias=True)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.init_weights(
            [self.input_gate, self.forget_gate, self.cell_gate, self.output_gate],
            "xavier"
        )

    @property
    def n_state_representations(self):
        return 2

    def update_state(self, inputs, states, ts=None):
        state, cell_state = states
        combined = torch.cat((inputs, state), dim=1)

        input_gate = self.sigmoid(self.input_gate(combined))
        forgat_gate = self.sigmoid(self.forget_gate(combined))
        cell_gate = self.tanh(self.cell_gate(combined))
        output_gate = self.sigmoid(self.output_gate(combined))

        cell_state = forgat_gate * cell_state + input_gate * cell_gate
        state = output_gate * torch.tanh(cell_gate)

        return state, (state, cell_state)

