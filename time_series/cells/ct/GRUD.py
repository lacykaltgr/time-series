# Copyright 2021 The ODE-LSTM Authors. All Rights Reserved.
import torch
from torch import nn

from time_series.cells.discrete.GRU import GRU


class GRUD(GRU):
    # Implemented according to
    # https://www.nature.com/articles/s41598-018-24271-9.pdf
    # without the masking

    def __init__(self, input_size, units, **args):
        super(GRUD, self).__init__(
            input_size=input_size,
            hidden_size=units,
            **args,
        )
        self._d_gate = nn.Linear(1, self.units, bias=True)
        self.init_weights([self._d_gate], "xavier")

    def update_state(self, inputs, state, elapsed):
        dt = self.relu(self._d_gate(elapsed))
        gamma = torch.exp(-dt)
        h_hat = state * gamma
        return super(GRUD, self).update_state(inputs, h_hat, elapsed)

