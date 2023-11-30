# Copyright 2021 The ODE-LSTM Authors. All Rights Reserved.
import torch
from torch import nn

from time_series.recurrent import Recurrent


class GRUD(Recurrent):
    # Implemented according to
    # https://www.nature.com/articles/s41598-018-24271-9.pdf
    # without the masking

    def __init__(self, input_size, units, **args):
        super(GRUD, self).__init__(
            input_size=input_size,
            units=units,
            **args,
        )

        self._reset_gate = nn.Linear(
            self.input_size + self.state_size, self.state_size, bias=True,
        )
        self._detect_signal = nn.Linear(
            self.input_size + self.state_size, self.state_size, bias=True,
        )
        self._update_gate = nn.Linear(
            self.input_size + self.state_size, self.state_size, bias=True,
        )
        self._d_gate = nn.Linear(
            1, self.units, bias=True,
        )
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.init_weights(
            [self._reset_gate, self._detect_signal, self._update_gate, self._d_gate],
            "xavier"
        )

    def update_state(self, inputs, state, elapsed):

        dt = self.relu(self._d_gate(elapsed))
        gamma = torch.exp(-dt)
        h_hat = state * gamma

        fused_input = torch.cat([inputs, h_hat], dim=-1)
        rt = self.sigmoid(self._reset_gate(fused_input))
        zt = self.sigmoid(self._update_gate(fused_input))

        reset_value = torch.cat([inputs, rt * h_hat], dim=-1)
        h_tilde = self.tanh(self._detect_signal(reset_value))

        ht = zt * h_hat + (1.0 - zt) * h_tilde
        return ht, ht
