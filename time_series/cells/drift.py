import torch

class DiffeqConcat(Module):
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
        activation: str,
        final_activation: str,
):
    super().__init__()
    self.net = st.net.MLP(dim + 1, hidden_dims, dim, activation, final_activation)

def forward(
        self,
        t: Tensor, # Time point, scalar
        state: Tuple[Tensor, Tensor]
) -> Tuple[Tensor, Tensor]:
    """ Input: t: (), state: tuple(x (..., n, d), diff (..., n, 1)) """
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

    def forward(
            self,
            t: Tensor,
            inp: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:

        h, diff = inp[0], inp[1]

        # Continuous gate functions
        r = torch.sigmoid(self.lin_hr(h))
        z = torch.sigmoid(self.lin_hz(h))
        u = torch.tanh(self.lin_hh(r * h))

        # Final drift
        dh = (1 - z) * (u - h) * diff

        return dh, torch.zeros_like(diff).to(dh)


#CTRNN
def dfdt(self, inputs, hidden_state):
    h_in = tf.matmul(inputs, self.kernel)
    h_rec = tf.matmul(hidden_state, self.recurrent_kernel)
    dh_in = self.scale * tf.nn.tanh(h_in + h_rec + self.bias)
    if self.tau > 0:
        dh = dh_in - hidden_state * self.tau
    else:
        dh = dh_in
    return dh


    self.kernel = self.add_weight(
        shape=(input_dim, self.units), initializer="glorot_uniform", name="kernel"
    )
    self.recurrent_kernel = self.add_weight(
        shape=(self.units, self.units),
        initializer="orthogonal",
        name="recurrent_kernel",
    )
    self.bias = self.add_weight(
        shape=(self.units), initializer=tf.keras.initializers.Zeros(), name="bias"
    )
    self.scale = self.add_weight(
        shape=(self.units),
        initializer=tf.keras.initializers.Constant(1.0),
        name="scale",
    )


