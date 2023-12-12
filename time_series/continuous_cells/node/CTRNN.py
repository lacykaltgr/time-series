# Copyright 2021 The ODE-LSTM Authors. All Rights Reserved.

import torch

class ODERNN(torch.nn.Module):
    def __init__(self, units, method, num_unfolds=None, tau=1, **kwargs):
        self.fixed_step_methods = {
            "euler": self.euler,
            "heun": self.heun,
            "rk4": self.rk4,
        }
        allowed_methods = ["euler", "heun", "rk4", "dopri5"]
        if not method in allowed_methods:
            raise ValueError(
                "Unknown ODE solver '{}', expected one of '{}'".format(
                    method, allowed_methods
                )
            )
        if method in self.fixed_step_methods.keys() and num_unfolds is None:
            raise ValueError(
                "Fixed-step ODE solver requires argument 'num_unfolds' to be specified!"
            )
        self.units = units
        self.state_size = units
        self.num_unfolds = num_unfolds
        self.method = method
        self.tau = tau
        super(CTRNNCell, self).__init__(**kwargs)


    class ODERNNEncoder(torch.nn.Module):
        def __init__(self, units, method, num_unfolds=None, tau=1, **kwargs):
            self.fixed_step_methods = {
                "euler": self.euler,
                "heun": self.heun,
                "rk4": self.rk4,
            }
            allowed_methods = ["euler", "heun", "rk4", "dopri5"]
            if not method in allowed_methods:
                raise ValueError(
                    "Unknown ODE solver '{}', expected one of '{}'".format(
                        method, allowed_methods
                    )
                )
            if method in self.fixed_step_methods.keys() and num_unfolds is None:
                raise ValueError(
                    "Fixed-step ODE solver requires argument 'num_unfolds' to be specified!"
                )
            self.units = units
            self.state_size = units
            self.num_unfolds = num_unfolds
            self.method = method
            self.tau = tau
            super(CTRNNCell, self).__init__(**kwargs)






