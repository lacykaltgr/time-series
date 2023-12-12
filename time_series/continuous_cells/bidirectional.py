# Copyright 2021 The ODE-LSTM Authors. All Rights Reserved.

import torch

class BidirectionalRNN(torch.nn.Module):
    def __init__(self, units, forward_rnn, backward_rnn, **kwargs):
        self.units = units
        self.state_size = (units, units, units)

        self.forward_rnn = forward_rnn
        self.backward_rnn = backward_rnn

        super(BidirectionalRNN, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if isinstance(input_shape[0], tuple):
            # Nested tuple
            input_dim = input_shape[0][-1]
        #self._out_layer = torch.nn.Linear(self.units)
        fused_dim = ((input_dim + self.units,), (1,))
        self.lstm.build(fused_dim)
        self.ctrnn.build(fused_dim)

    def forward(self, inputs, states):
        elapsed = 1.0
        if (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) > 1:
            elapsed = inputs[1]
            inputs = inputs[0]

        lstm_state = [states[0], states[1]]
        lstm_input = [torch.concat([inputs, states[2]], dim=-1), elapsed]
        ctrnn_state = [states[2]]
        ctrnn_input = [torch.concat([inputs, states[1]], dim=-1), elapsed]

        lstm_out, new_lstm_states = self.lstm(lstm_input, lstm_state)
        ctrnn_out, new_ctrnn_state = self.ctrnn(ctrnn_input, ctrnn_state)

        fused_output = lstm_out + ctrnn_out
        return (
            fused_output,
            [new_lstm_states[0], new_lstm_states[1], new_ctrnn_state[0]],
        )
