# Copyright 2021 The ODE-LSTM Authors. All Rights Reserved.

from torch import nn
import torch
import numpy as np

from time_series.recurrent import Recurrent


class CTGRU(Recurrent):
    # https://arxiv.org/abs/1710.04110
    def __init__(self, input_size, units: int, M=8, **args):
        super(CTGRU, self).__init__(
            input_size=input_size,
            units=units * M,
            **args
        )
        self.units = units
        self.M = M

        # Pre-computed tau table (as recommended in paper)
        self.ln_tau_table = np.empty(self.M)
        self.tau_table = np.empty(self.M)
        tau = 1.0
        for i in range(self.M):
            self.ln_tau_table[i] = np.log(tau)
            self.tau_table[i] = tau
            tau = tau * (10.0 ** 0.5)

        self.retrieval_layer = nn.Linear(
            self.input_size + self.units, self.state_size, bias=True)
        self.detect_layer = nn.Linear(
            self.input_size + self.units, self.units, bias=True)
        self.update_layer = nn.Linear(
            self.input_size + self.units, self.state_size, bias=True)
        self.tanh = nn.Tanh()

        self.init_weights(
            [self.retrieval_layer, self.update_layer, self.detect_layer],
            "xavier"
        )

    def update_state(self, inputs, state, elapsed):
        # States is actually 2D
        h_hat = state.reshape([..., self.units, self.M])
        h = torch.sum(h_hat, dim=2)

        # Retrieval
        fused_input = torch.cat([inputs, h], dim=-1)
        ln_tau_r = self.retrieval_layer(fused_input)
        ln_tau_r = ln_tau_r.reshape([..., self.units, self.M])
        sf_input_r = -torch.square(ln_tau_r - self.ln_tau_table)
        rki = torch.nn.Softmax(dim=2)(sf_input_r)

        q_input = torch.sum(rki * h_hat, dim=2)
        reset_value = torch.cat([inputs, q_input], dim=1)
        qk = self.tanh(self.detect_layer(reset_value))
        qk = qk.reshape([..., self.units, 1])  # in order to broadcast

        ln_tau_s = self.update_layer(fused_input)
        ln_tau_s = ln_tau_s.reshape([..., self.units, self.M])
        sf_input_s = -torch.square(ln_tau_s - self.ln_tau_table)
        ski = torch.nn.Softmax(dim=2)(sf_input_s)

        # Now the elapsed time enters the state update
        base_term = (1 - ski) * h_hat + ski * qk
        exp_term = torch.exp(-elapsed / self.tau_table)
        exp_term = exp_term.reshape([..., 1, self.M])
        h_hat_next = base_term * exp_term

        # Compute new state
        h_next = torch.sum(h_hat_next, dim=2)
        h_hat_next_flat = h_hat_next.reshape([..., self.units * self.M])

        return h_next, h_hat_next_flat
