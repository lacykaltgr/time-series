# Copyright 2021 The ODE-LSTM Authors. All Rights Reserved.
import torch
from torch import nn

from time_series.recurrent import Recurrent


class HawkLSTM(Recurrent):
    # https://papers.nips.cc/paper/7252-the-neural-hawkes-process-a-neurally-self-modulating-multivariate-point-process.pdf
    def __init__(self, input_size, units, **args):
        super(HawkLSTM, self).__init__(
            input_size=input_size,
            units=units,
            **args,
        )

        self.input_kernel = nn.Linear(self.input_size, 7 * self.units, bias=True)
        self.recurrent_kernel = nn.Linear(self.units, 7 * self.units)
        self.init_weights([self.input_kernel], "xavier")
        self.init_weights([self.recurrent_kernel], "orthogonal")

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    @property
    def n_state_representations(self):
        return 3

    def update_state(self, inputs, states, elapsed):
        c, c_bar, h = states
        z = self.input_kernel(inputs) + self.recurrent_kernel(h)
        i, ig, fg, og, ig_bar, fg_bar, d = torch.split(z, 7, dim=-1)

        input_activation = self.tanh(i)
        input_gate = self.sigmoid(ig)
        input_gate_bar = self.sigmoid(ig_bar)
        forget_gate = self.sigmoid(fg)
        forget_gate_bar = self.sigmoid(fg_bar)
        output_gate = self.sigmoid(og)
        delta_gate = self.softplus(d)

        new_c = c * forget_gate + input_activation * input_gate
        new_c_bar = c_bar * forget_gate_bar + input_activation * input_gate_bar

        c_t = new_c_bar + (new_c - new_c_bar) * torch.exp(-delta_gate * elapsed)
        output_state = self.tanh(c_t) * output_gate

        return output_state, (new_c, new_c_bar, output_state)
