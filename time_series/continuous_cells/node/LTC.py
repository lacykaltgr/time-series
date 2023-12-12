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

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union

from time_series.continuous_cells.ode import BaseNODE


class LTC(BaseNODE):
    def __init__(
            self,
            input_size,
            units,

            solver_step=6,
            implicit_param_constraints=False,
            **kwargs,
    ):
        """A `Liquid time-constant (LTC) <https://ojs.aaai.org/index.php/AAAI/article/view/16936>`_ cell.

        :param wiring:
        :param in_features:
        :param input_mapping:
        :param output_mapping:
        :param ode_unfolds:
        :param epsilon:
        :param implicit_param_constraints:
        """

        super().__init__(
            input_size,
            units,
            custom_solver= self.solver_function,
            solver_step=solver_step,
            **kwargs
        )

        self.make_positive_fn = (nn.Softplus() if implicit_param_constraints else nn.Identity())
        self._implicit_param_constraints = implicit_param_constraints
        self._solver_step = solver_step
        self._clip = torch.nn.ReLU()

        self.gleak = nn.Parameter(self._get_init_value((self.state_size,), "gleak"), requires_grad=True)
        self.vleak = nn.Parameter(self._get_init_value((self.state_size,), "vleak"), requires_grad=True)
        self.cm = nn.Parameter(self._get_init_value((self.state_size,), "cm"), requires_grad=True)
        self.sigma = nn.Parameter(self._get_init_value((self.state_size, self.state_size), "sigma"), requires_grad=True)
        self.mu = nn.Parameter(self._get_init_value((self.state_size, self.state_size), "mu"), requires_grad=True)
        self.w = nn.Parameter(self._get_init_value((self.state_size, self.state_size), "w"), requires_grad=True)
        self.erev = nn.Parameter(self._get_init_value((self.state_size, self.state_size), "erev"), requires_grad=True)
        self.sensory_sigma = nn.Parameter(self._get_init_value((self.sensory_size, self.state_size), "sensory_sigma"), requires_grad=True)
        self.sensory_mu = nn.Parameter(self._get_init_value((self.sensory_size, self.state_size), "sensory_mu"), requires_grad=True)
        self.sensory_w = nn.Parameter(self._get_init_value((self.sensory_size, self.state_size), "sensory_w"), requires_grad=True)
        self.sensory_erev = nn.Parameter(self._get_init_value((self.sensory_size, self.state_size), "sensory_erev"), requires_grad=True)
        self.sparsity_mask = nn.Parameter(self._get_init_value((self.state_size, self.state_size), "sparsity_mask"), requires_grad=False)
        self.sensory_sparsity_mask = nn.Parameter(self._get_init_value((self.sensory_size, self.state_size), "sensory_sparsity_mask"), requires_grad=False)

    def _get_init_value(self, shape, param_name):
        _init_ranges = {
            "gleak": (0.001, 1.0),
            "vleak": (-0.2, 0.2),
            "cm": (0.4, 0.6),
            "w": (0.001, 1.0),
            "sigma": (3, 8),
            "mu": (0.3, 0.8),
            "sensory_w": (0.001, 1.0),
            "sensory_sigma": (3, 8),
            "sensory_mu": (0.3, 0.8),
        }
        minval, maxval = _init_ranges[param_name]
        if minval == maxval:
            return torch.ones(shape) * minval
        else:
            return torch.rand(*shape) * (maxval - minval) + minval

    def _ode_solver(self, inputs, state, elapsed_time):
        def _sigmoid(v_pre, mu, sigma):
            v_pre = torch.unsqueeze(v_pre, -1)  # For broadcasting
            mues = v_pre - mu
            x = sigma * mues
            return torch.sigmoid(x)

        v_pre = state

        # We can pre-compute the effects of the sensory neurons here
        sensory_w_activation = self.make_positive_fn(
            self._params["sensory_w"]
        ) * _sigmoid(
            inputs, self._params["sensory_mu"], self._params["sensory_sigma"]
        )
        sensory_w_activation = (
                sensory_w_activation * self._params["sensory_sparsity_mask"]
        )

        sensory_rev_activation = sensory_w_activation * self._params["sensory_erev"]

        # Reduce over dimension 1 (=source sensory neurons)
        w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_w_activation, dim=1)

        # cm/t is loop invariant
        cm_t = self.make_positive_fn(self._params["cm"]) / (
                elapsed_time / self._ode_unfolds
        )

        # Unfold the multiply ODE multiple times into one RNN step
        w_param = self.make_positive_fn(self._params["w"])
        for t in range(self._ode_unfolds):
            w_activation = w_param * self._sigmoid(
                v_pre, self._params["mu"], self._params["sigma"]
            )

            w_activation = w_activation * self._params["sparsity_mask"]

            rev_activation = w_activation * self._params["erev"]

            # Reduce over dimension 1 (=source neurons)
            w_numerator = torch.sum(rev_activation, dim=1) + w_numerator_sensory
            w_denominator = torch.sum(w_activation, dim=1) + w_denominator_sensory

            gleak = self.make_positive_fn(self._params["gleak"])
            numerator = cm_t * v_pre + gleak * self._params["vleak"] + w_numerator
            denominator = cm_t + gleak + w_denominator

            # Avoid dividing by 0
            v_pre = numerator / (denominator + self._epsilon)

        return v_pre

    def apply_weight_constraints(self):
        if not self._implicit_param_constraints:
            # In implicit mode, the parameter constraints are implemented via
            # a softplus function at runtime
            self._params["w"].data = self._clip(self._params["w"].data)
            self._params["sensory_w"].data = self._clip(self._params["sensory_w"].data)
            self._params["cm"].data = self._clip(self._params["cm"].data)
            self._params["gleak"].data = self._clip(self._params["gleak"].data)

