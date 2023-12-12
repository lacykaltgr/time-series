# Copyright 2022 Mathias Lechner and Ramin Hasani
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union
import numpy as np
import torch
from torch import nn
from wiring import Wiring, FullyConnected
from abc import ABC, abstractmethod


class Recurrent(nn.Module, ABC):
    def __init__(
            self,
            input_size: int,
            units: Union[int, Wiring],
            return_sequences: bool = True,
            batch_first: bool = True,
    ):
        """
        :param input_size: Number of input features
        :param units: Wiring (ncps.wirings.Wiring instance) or integer representing the number of (fully-connected) hidden units
        :param return_sequences: Whether to return the full sequence or just the last output
        :param batch_first: Whether the batch or time dimension is the first (0-th) dimension
        """

        super(Recurrent, self).__init__()
        self._input_size = input_size
        self.batch_first = batch_first
        self.return_sequences = return_sequences

        if isinstance(units, Wiring):
            wiring = units
        else:
            wiring = FullyConnected(units)
        self._wiring = wiring

    @property
    def state_size(self):
        return self._wiring.units

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._wiring.output_dim

    @property
    def synapse_count(self):
        return np.sum(np.abs(self._wiring.adjacency_matrix))

    @staticmethod
    def init_weights(modules, initializer_scheme):
        initializer_schemes = dict(
            uniform=lambda w: torch.nn.init.uniform_(w, -0.1, 0.1),
            xavier=lambda w: torch.nn.init.xavier_uniform_(w),
            orthogonal=lambda w: torch.nn.init.orthogonal_(w)
        )
        if initializer_scheme not in initializer_schemes.keys():
            raise ValueError(
                f"Initializer {initializer_scheme} not available!"
                f"Please choose one of these: {initializer_schemes.keys()}"
            )

        for module in modules:
            for w in module.parameters():
                if w.dim() == 1:
                    initializer_scheme["uniform"](w)
                else:
                    initializer_scheme[initializer_scheme](w)

    @abstractmethod
    def update_state(self, inputs, states, ts):
        pass

    @property
    def n_state_representations(self):
        return 1

    def init_state(self, hx, batch_size, is_batched, device):
        if hx is None:
            state = ()
            for i in range(self.n_state_representations):
                state_repr = torch.zeros((batch_size, self.state_size), device=device)
                state = state + (state_repr, )

        else:
            if self.n_state_representations > 1 and isinstance(hx, torch.Tensor):
                raise RuntimeError(
                    "Initializing multiple state representations requires a tuple (h_a, h_b, ...) "
                    "to be passed as state (got torch.Tensor instead)"
                )
            if len(hx) != self.n_state_representations:
                raise RuntimeError(
                    "Initializing multiple state representations "
                    "requires a tuple of the same lenght as the number of state representations"
                )
            state = hx
            state_repr1 = state[0]
            if is_batched:
                if state_repr1.dim() != 2:
                    raise RuntimeError(
                        "For batched 2-D input, state representations should "
                        f"also be 2-D but got ({state_repr1.dim()}-D) tensor"
                    )
            else:
                # batchless  mode
                if state_repr1.dim() != 1:
                    raise RuntimeError(
                        "For unbatched 1-D input, state representations should "
                        f"also be 1-D but got ({state_repr1.dim()}-D) tensor")
                for i in range(len(state)):
                    state[i] = state[i].unsqueeze(0)
        return state

    def forward(self, input, hx=None, timespans=None):
        """
        :param input: Input tensor of shape
            (L,C) in batchless mode,
            or (B,L,C) if batch_first was set to True
            and (L,B,C) if batch_first is False
        :param hx: Initial hidden state of the RNN of shape (B,H)
                If None, the hidden states are initialized with all zeros.
        :param timespans: Optional timespan tensor
        :return: A pair (output, hx), where output and hx the final hidden state of the RNN
        """
        device = input.device
        is_batched = input.dim() == 3
        batch_dim = 0 if self.batch_first else 1
        seq_dim = 1 if self.batch_first else 0
        if not is_batched:
            input = input.unsqueeze(batch_dim)
            if timespans is not None:
                timespans = timespans.unsqueeze(batch_dim)
        batch_size, seq_len = input.size(batch_dim), input.size(seq_dim)

        state = self.init_state(hx, batch_size, is_batched, device)
        output_sequence = []
        for t in range(seq_len):
            if self.batch_first:
                inputs = input[:, t]
                ts = 1.0 if timespans is None else timespans[:, t].squeeze()
            else:
                inputs = input[t]
                ts = 1.0 if timespans is None else timespans[t].squeeze()

            output, state = self.update_state(inputs, state, ts)

            if self.return_sequences:
                output_sequence.append(output)

        if self.return_sequences:
            stack_dim = 1 if self.batch_first else 0
            readout = torch.stack(output_sequence, dim=stack_dim)
        else:
            readout = output

        return readout, state


    def _map_inputs(self, inputs):
        if self._input_mapping in ["affine", "linear"]:
            inputs = inputs * self._params["input_w"]
        if self._input_mapping == "affine":
            inputs = inputs + self._params["input_b"]
        return inputs

    def _map_outputs(self, state):
        output = state
        if self.motor_size < self.state_size:
            output = output[:, 0 : self.motor_size]  # slice

        if self._output_mapping in ["affine", "linear"]:
            output = output * self._params["output_w"]
        if self._output_mapping == "affine":
            output = output + self._params["output_b"]
        return output

input_mapping="affine",
output_mapping="affine",

if self._input_mapping in ["affine", "linear"]:
    self._params["input_w"] = self.add_weight(
        name="input_w",
        init_value=torch.ones((self.sensory_size,)),
    )
if self._input_mapping == "affine":
    self._params["input_b"] = self.add_weight(
        name="input_b",
        init_value=torch.zeros((self.sensory_size,)),
    )

if self._output_mapping in ["affine", "linear"]:
    self._params["output_w"] = self.add_weight(
        name="output_w",
        init_value=torch.ones((self.motor_size,)),
    )
if self._output_mapping == "affine":
    self._params["output_b"] = self.add_weight(
        name="output_b",
        init_value=torch.zeros((self.motor_size,)),
    )