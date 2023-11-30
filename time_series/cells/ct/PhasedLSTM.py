# Copyright 2021 The ODE-LSTM Authors. All Rights Reserved.
from torch import nn
import torch

from time_series.recurrent import Recurrent


class PhasedLSTM(Recurrent):
    # Implemented according to
    # https://papers.nips.cc/paper/6310-phased-lstm-accelerating-recurrent-network-training-for-long-or-event-based-sequences.pdf

    def __init__(self, input_size, units, **args):
        super(PhasedLSTM, self).__init__(
            input_size=input_size,
            units=units,
            **args,
        )

        self.input_kernel = nn.Linear(self.input_size, 4 * self.units, bias=True)
        self.recurrent_kernel = nn.Linear(self.units, 4 * self.units, bias=False)

        self.init_weights([self.input_kernel], "xavier")
        self.init_weights([self.recurrent_kernel], "orthogonal")

        self.tau = nn.Parameter(torch.zeros((1,)), requires_grad=True)
        self.ron = nn.Parameter(torch.zeros((1,)), requires_grad=True)
        self.s = nn.Parameter(torch.zeros((1,)), requires_grad=True)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    @property
    def n_state_representations(self):
        return 2

    def update_state(self, inputs, states, elapsed):
        cell_state, hidden_state = states

        # Leaky constant taken fromt he paper
        alpha = 0.001
        # Make sure these values are positive
        tau = self.softplus(self.tau)
        s = self.softplus(self.s)
        ron = self.softplus(self.ron)

        phit = torch.fmod(elapsed - s, tau) / tau
        kt = torch.where(
            torch.less(phit, 0.5 * ron),
            2 * phit * ron,
            torch.where(
                torch.less(phit, ron),
                2.0 - 2 * phit / ron,
                alpha * phit),
        )

        z = self.input_kernel(inputs) + self.recurrent_kernel(hidden_state)
        i, ig, fg, og = torch.split(z, 4, dim=-1)

        input_activation = self.tanh(i)
        input_gate = self.sigmoid(ig)
        forget_gate = self.sigmoid(fg + 1.0)
        output_gate = self.sigmoid(og)

        c_tilde = cell_state * forget_gate + input_activation * input_gate
        c = kt * c_tilde + (1.0 - kt) * cell_state

        h_tilde = self.tanh(c_tilde) * output_gate
        h = kt * h_tilde + (1.0 - kt) * hidden_state

        return h, (c, h)