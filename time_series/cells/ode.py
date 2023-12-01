from typing import List, Optional, Tuple, Union

import torch
from torch.nn import Module
from torchdiffeq import odeint_adjoint as odeint

from time_series.recurrent import Recurrent
from time_series.cells.drift import DiffeqConcat


class BaseNODE(Recurrent):
    """
    Neural ordinary differential equation model
    Implements reparameterization and seminorm trick for ODEs

    Args:
        dim: Data dimension
        net: Either a name (only `concat` supported) or a torch.Module
        hidden_dims: Hidden dimensions of the neural network
        activation: Name of the activation function from `torch.nn`
        final_activation: Name of the activation function from `torch.nn`
        solver: Which numerical solver to use (e.g. `dopri5`, `euler`, `rk4`)
        solver_step: How many solvers steps to take, only applicable for fixed step solvers
        atol: Absolute tolerance
        rtol: Relative tolerance
    """

    def __init__(
            self,
            input_size: int,
            units: int,
            hidden_dims: List[int],
            drift: Union[str, Module],

            solver: str,
            solver_step: Optional[int] = None,
            atol: Optional[float] = 1e-4,
            rtol: Optional[float] = 1e-3,
            **kwargs
    ):
        super().__init__(
            input_size=input_size,
            units=units,
            **kwargs
        )

        self.atol = atol
        self.rtol = rtol

        if drift == 'concat':
            self.net = DiffeqConcat(input_size, hidden_dims)
        elif isinstance(drift, Module):
            self.net = drift
        else:
            raise NotImplementedError

        self.solver = solver

        if solver == 'dopri5':
            self.options = None
        else:
            self.options = {'step_size': solver_step}

    def update_state(self, inputs, states, ts):
        state = odeint(
            self.net,  # Drift network
            (x, ts),  # Initial condition
            torch.Tensor([0, 1]).to(x),  # Reparameterization trick
            method=self.solver,
            options=self.options,
            atol=self.atol,
            rtol=self.rtol,
            adjoint_options=dict(norm='seminorm')  # Seminorm trick
        )[0][1]  # get first state (x), second output (at t=1)

        return state, state,

    def euler(self, inputs, hidden_state, delta_t):
        dy = self.dfdt(inputs, hidden_state)
        return hidden_state + delta_t * dy

    def heun(self, inputs, hidden_state, delta_t):
        k1 = self.dfdt(inputs, hidden_state)
        k2 = self.dfdt(inputs, hidden_state + delta_t * k1)
        return hidden_state + delta_t * 0.5 * (k1 + k2)

    def rk4(self, inputs, hidden_state, delta_t):
        k1 = self.dfdt(inputs, hidden_state)
        k2 = self.dfdt(inputs, hidden_state + k1 * delta_t * 0.5)
        k3 = self.dfdt(inputs, hidden_state + k2 * delta_t * 0.5)
        k4 = self.dfdt(inputs, hidden_state + k3 * delta_t)

        return hidden_state + delta_t * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


# for ODELSTM, GRULSTM
class BaseEncoderNODE(BaseNODE):
    def __init__(
            self,
            input_size: int,
            units: int,
            hidden_dims: List[int],
            drift: Union[str, Module],

            solver: str,
            solver_step: Optional[int] = None,
            atol: Optional[float] = 1e-4,
            rtol: Optional[float] = 1e-3,
    ):
        super().__init__(
            input_size=input_size,
            units=units,
            hidden_dims=hidden_dims,
            drift=drift,
            solver=solver,
            solver_step=solver_step,
            atol=atol,
            rtol=rtol,
        )

    def update_state(self, x, ts):
        return odeint(
            self.net,  # Drift network
            (x, ts),  # Initial condition
            torch.Tensor([0, 1]).to(x),  # Reparameterization trick
            method=self.solver,
            options=self.options,
            atol=self.atol,
            rtol=self.rtol,
            adjoint_options=dict(norm='seminorm')  # Seminorm trick
        )[0][1]  # get first state (x), second output (at t=1)


class BaserLTCNODE(BaseNODE):
    def __init__(
                self,
                input_size: int,
                units: int,
                hidden_dims: List[int],
                drift: Union[str, Module],

                solver: str,
                solver_step: Optional[int] = None,
                atol: Optional[float] = 1e-4,
                rtol: Optional[float] = 1e-3,
        ):
            super().__init__(
                input_size=input_size,
                units=units,
                hidden_dims=hidden_dims,
                drift=drift,
                solver=solver,
                solver_step=solver_step,
                atol=atol,
                rtol=rtol,
            )

    def _sigmoid(self, v_pre, mu, sigma):
            v_pre = torch.unsqueeze(v_pre, -1)  # For broadcasting
            mues = v_pre - mu
            x = sigma * mues
            return torch.sigmoid(x)

    def _ode_solver(self, inputs, state, elapsed_time):
        v_pre = state

        # We can pre-compute the effects of the sensory neurons here
        sensory_w_activation = self.make_positive_fn(
            self._params["sensory_w"]
        ) * self._sigmoid(
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


class BaseCfCNODE(BaseNODE):
    def __init__(
            self,
            input_size: int,
            units: int,
            hidden_dims: List[int],
            drift: Union[str, Module],

            solver: str,
            solver_step: Optional[int] = None,
            atol: Optional[float] = 1e-4,
            rtol: Optional[float] = 1e-3,
    ):
        super().__init__(
            input_size=input_size,
            units=units,
            hidden_dims=hidden_dims,
            drift=drift,
            solver=solver,
            solver_step=solver_step,
            atol=atol,
            rtol=rtol,
        )

    def update_state(self, x, ts):
        return odeint(
            self.net,  # Drift network
            (x, ts),  # Initial condition
            torch.Tensor([0, 1]).to(x),  # Reparameterization trick
            method=self.solver,
            options=self.options,
            atol=self.atol,
            rtol=self.rtol,
            adjoint_options=dict(norm='seminorm')  # Seminorm trick
        )[0][1]  # get first state (x), second output (at t=1)
