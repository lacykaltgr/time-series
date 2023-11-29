# Copyright 2021 The ODE-LSTM Authors. All Rights Reserved.


class GRUD(tf.keras.layers.Layer):
    # Implemented according to
    # https://www.nature.com/articles/s41598-018-24271-9.pdf
    # without the masking

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(GRUD, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if isinstance(input_shape[0], tuple):
            # Nested tuple
            input_dim = input_shape[0][-1]

        self._reset_gate = tf.keras.layers.Dense(
            self.units, activation="sigmoid", kernel_initializer="glorot_uniform"
        )
        self._detect_signal = tf.keras.layers.Dense(
            self.units, activation="tanh", kernel_initializer="glorot_uniform"
        )
        self._update_gate = tf.keras.layers.Dense(
            self.units, activation="sigmoid", kernel_initializer="glorot_uniform"
        )
        self._d_gate = tf.keras.layers.Dense(
            self.units, activation="relu", kernel_initializer="glorot_uniform"
        )

        self.built = True

    def call(self, inputs, states):
        elapsed = 1.0
        if (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) > 1:
            elapsed = inputs[1]
            inputs = inputs[0]

        dt = self._d_gate(elapsed)
        gamma = tf.exp(-dt)
        h_hat = states[0] * gamma

        fused_input = tf.concat([inputs, h_hat], axis=-1)
        rt = self._reset_gate(fused_input)
        zt = self._update_gate(fused_input)

        reset_value = tf.concat([inputs, rt * h_hat], axis=-1)
        h_tilde = self._detect_signal(reset_value)

        # Compute new state
        ht = zt * h_hat + (1.0 - zt) * h_tilde

        return ht, [ht]