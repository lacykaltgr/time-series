from time_series.continuous_cells.ode import BaseNODE
from time_series.continuous_cells.ode import GRUDrift

# ODE with gru drift as drift function
class GRUODE(BaseNODE):
    def __init__(self, input_size, units, **kwargs):
        super(GRUODE, self).__init__(
            input_size,
            units,
            drift=GRUDrift(units),
            **kwargs
        )