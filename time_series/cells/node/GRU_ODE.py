# Copyright 2021 The ODE-LSTM Authors. All Rights Reserved.


class GRUODE(tf.keras.layers.Layer):
    # Implemented according to
    # https://arxiv.org/pdf/1905.12374.pdf
    # without the Bayesian stuff

    def __init__(self, units, num_unfolds=4, **kwargs):
        self.units = units
        self.num_unfolds = num_unfolds
        self.state_size = units
        super(GRUODE, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if isinstance(input_shape[0], tuple):
            # Nested tuple
            input_dim = input_shape[0][-1]
        self._reset_gate = tf.keras.layers.Dense(
            self.units,
            activation="sigmoid",
            bias_initializer=tf.constant_initializer(1),
        )
        self._detect_signal = tf.keras.layers.Dense(self.units, activation="tanh")
        self._update_gate = tf.keras.layers.Dense(self.units, activation="sigmoid")

        self.built = True

    def _dh_dt(self, inputs, states):
        fused_input = tf.concat([inputs, states], axis=-1)
        rt = self._reset_gate(fused_input)
        zt = self._update_gate(fused_input)

        reset_value = tf.concat([inputs, rt * states], axis=-1)
        gt = self._detect_signal(reset_value)

        # Compute new state
        dhdt = (1.0 - zt) * (gt - states)
        return dhdt

    def euler(self, inputs, hidden_state, delta_t):
        dy = self._dh_dt(inputs, hidden_state)
        return hidden_state + delta_t * dy

    def call(self, inputs, states):
        elapsed = 1.0
        if (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) > 1:
            elapsed = inputs[1]
            inputs = inputs[0]

        delta_t = elapsed / self.num_unfolds
        hidden_state = states[0]
        for i in range(self.num_unfolds):
            hidden_state = self.euler(inputs, hidden_state, delta_t)
        return hidden_state, [hidden_state]

        return ht, [ht]



class GRUODENet(Module):
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