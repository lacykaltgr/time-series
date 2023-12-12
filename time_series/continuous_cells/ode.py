from typing import Union, Callable

from torch.nn import Module
from torchdiffeq import odeint_adjoint as odeint

from time_series.recurrent import Recurrent

import torch
from torch import nn

from typing import List, Optional



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
            drift: Union[str, Module],

            solver: str,
            custom_solver: Optional[Callable] = None,
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

        if isinstance(drift, nn.Module):
            self.net = drift
        else:
            raise NotImplementedError

        if custom_solver is not None:
            self.solver_function = custom_solver
        elif solver is not None:
            if solver == 'dopri5':
                self.options = None
            else:
                self.options = {'step_size': solver_step}
            solver = solver
            self.solver_function = self.ode_solve
        else:
            raise ValueError('Either `solver` or `custom_solver` must be specified')




    def update_state(self, inputs, state, ts):
        new_state = self.solver_function()
        return new_state, new_state


    def ode_solve(self, inputs, state, ts):
        new_state = odeint(
            func=self.net,                          # Drift network
            y0=(state, ts),                        # Initial condition
            t=torch.Tensor([0, 1]).to(state),      # Reparameterization trick
            method=self.solver,
            options=self.options,
            atol=self.atol,
            rtol=self.rtol,
            adjoint_options=dict(norm='seminorm')  # Seminorm trick
        )                                          # get first state (x), second output (at t=1)
        return new_state, new_state

class DiffeqConcat(nn.Module):
    """
    Drift function for neural ODE model

    Args:
        dim: Data dimension
        hidden_dims: Hidden dimensions of the neural network
        activation: Name of the activation function from `torch.nn`
        final_activation: Name of the activation function from `torch.nn`
    """
    def __init__(
            self,
            dim: int,
            hidden_dims: List[int],
            activation: nn.Module,
            final_activation: nn.Module,
    ):
        super().__init__()
        dims = [dim + 1] + hidden_dims + [dim]
        self.net = nn.Sequential(*[
            nn.Sequential(nn.Linear(dims[i], dims[i + 1]), activation)
                if i < len(dims) - 1 else nn.Sequential(nn.Linear(dims[i], dims[i + 1], bias=False), final_activation)
            for i in range(len(dims))
        ])

    def forward(self, t, state):
        """ t: (), state: tuple(x (..., n, d), diff (..., n, 1))"""
        x, diff = state
        x = torch.cat([t * diff, x], -1)
        dx = self.net(x) * diff
        return dx, torch.zeros_like(diff).to(dx)


class GRUDrift(Module):
    """
    GRU-ODE drift function

    Args:
        hidden_dim: Size of the GRU hidden state
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.lin_hh = nn.Linear(hidden_dim, hidden_dim)
        self.lin_hz = nn.Linear(hidden_dim, hidden_dim)
        self.lin_hr = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, t, state):
        h, diff = state

        # Continuous gate functions
        r = torch.sigmoid(self.lin_hr(h))
        z = torch.sigmoid(self.lin_hz(h))
        u = torch.tanh(self.lin_hh(r * h))

        # Final drift
        dh = (1 - z) * (u - h) * diff
        return dh, torch.zeros_like(diff).to(dh)




    def _f_prime(self,inputs,state):
        v_pre = state

        # We can pre-compute the effects of the sensory neurons here
        sensory_w_activation = self.sensory_W*self._sigmoid(inputs,self.sensory_mu,self.sensory_sigma)
        w_reduced_sensory = tf.reduce_sum(sensory_w_activation,axis=1)

        # Unfold the mutliply ODE multiple times into one RNN step
        w_activation = self.W*self._sigmoid(v_pre,self.mu,self.sigma)

        w_reduced_synapse = tf.reduce_sum(w_activation,axis=1)

        sensory_in = self.sensory_erev * sensory_w_activation
        synapse_in = self.erev * w_activation

        sum_in = tf.reduce_sum(sensory_in,axis=1) - v_pre*w_reduced_synapse + tf.reduce_sum(synapse_in,axis=1) - v_pre * w_reduced_sensory

        f_prime = 1/self.cm_t * (self.gleak * (self.vleak-v_pre) + sum_in)

        return f_prime

    def _ode_step_runge_kutta(self,inputs,state):

        h = 0.1
        for i in range(self._ode_solver_unfolds):
            k1 = h*self._f_prime(inputs,state)
            k2 = h*self._f_prime(inputs,state+k1*0.5)
            k3 = h*self._f_prime(inputs,state+k2*0.5)
            k4 = h*self._f_prime(inputs,state+k3)

            state = state + 1.0/6*(k1+2*k2+2*k3+k4)

        return state

    def _ode_step_explicit(self,inputs,state,_ode_solver_unfolds):
        v_pre = state

        # We can pre-compute the effects of the sensory neurons here
        sensory_w_activation = self.sensory_W*self._sigmoid(inputs,self.sensory_mu,self.sensory_sigma)
        w_reduced_sensory = tf.reduce_sum(sensory_w_activation,axis=1)


        # Unfold the mutliply ODE multiple times into one RNN step
        for t in range(_ode_solver_unfolds):
            w_activation = self.W*self._sigmoid(v_pre,self.mu,self.sigma)

            w_reduced_synapse = tf.reduce_sum(w_activation,axis=1)

            sensory_in = self.sensory_erev * sensory_w_activation
            synapse_in = self.erev * w_activation

            sum_in = tf.reduce_sum(sensory_in,axis=1) - v_pre*w_reduced_synapse + tf.reduce_sum(synapse_in,axis=1) - v_pre * w_reduced_sensory

            f_prime = 1/self.cm_t * (self.gleak * (self.vleak-v_pre) + sum_in)

            v_pre = v_pre + 0.1 * f_prime

        return v_pre

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






