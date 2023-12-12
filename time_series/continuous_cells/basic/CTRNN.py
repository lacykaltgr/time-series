# Copyright 2021 The ODE-LSTM Authors. All Rights Reserved.

from torch import nn
import torch

from time_series.recurrent import Recurrent


class CTRNN(Recurrent):
    def __init__(self, input_size, units, **args):
        super(CTRNN, self).__init__(
            input_size=input_size,
            units=units,
            **args,
        )
        self.layer = nn.Linear(self.input_size, self.state_size, bias=True)
        self.out_layer = nn.Linear(self.state_size, self.state_size, bias=False)
        self.tanh = nn.Tanh()
        self.tau = nn.Parameter(torch.ones(self.units) * 0.1, requires_grad=True)

    def update_state(self, inputs, state, elapsed):
        fused_input = torch.cat([inputs, state], dim=-1)
        new_states = self.out_layer(self.layer(fused_input)) - elapsed * self._tau
        return new_states, new_states

"""
        def func_dfdt(self, inputs, hidden_state):
            h_in = tf.matmul(inputs, self.kernel)
        h_rec = tf.matmul(hidden_state, self.recurrent_kernel)
        dh_in = self.scale * tf.nn.tanh(h_in + h_rec + self.bias)
        if self.tau > 0:
            dh = dh_in - hidden_state * self.tau
        else:
            dh = dh_in
        return dh      
"""


