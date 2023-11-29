# Copyright 2021 The ODE-LSTM Authors. All Rights Reserved.

class PhasedLSTM(tf.keras.layers.Layer):
    # Implemented according to
    # https://papers.nips.cc/paper/6310-phased-lstm-accelerating-recurrent-network-training-for-long-or-event-based-sequences.pdf

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = (units, units)
        self.initializer = "glorot_uniform"
        self.recurrent_initializer = "orthogonal"
        super(PhasedLSTM, self).__init__(**kwargs)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return (
            tf.zeros([batch_size, self.units], dtype=tf.float32),
            tf.zeros([batch_size, self.units], dtype=tf.float32),
        )

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if isinstance(input_shape[0], tuple):
            # Nested tuple
            input_dim = input_shape[0][-1]

        self.input_kernel = self.add_weight(
            shape=(input_dim, 4 * self.units),
            initializer=self.initializer,
            name="input_kernel",
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, 4 * self.units),
            initializer=self.recurrent_initializer,
            name="recurrent_kernel",
        )
        self.bias = self.add_weight(
            shape=(4 * self.units),
            initializer=tf.keras.initializers.Zeros(),
            name="bias",
        )
        self.tau = self.add_weight(
            shape=(1,), initializer=tf.keras.initializers.Zeros(), name="tau"
        )
        self.ron = self.add_weight(
            shape=(1,), initializer=tf.keras.initializers.Zeros(), name="ron"
        )
        self.s = self.add_weight(
            shape=(1,), initializer=tf.keras.initializers.Zeros(), name="s"
        )

        self.built = True

    def call(self, inputs, states):
        cell_state, hidden_state = states
        elapsed = 1.0
        if (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) > 1:
            elapsed = inputs[1]
            inputs = inputs[0]

        # Leaky constant taken fromt he paper
        alpha = 0.001
        # Make sure these values are positive
        tau = tf.nn.softplus(self.tau)
        s = tf.nn.softplus(self.s)
        ron = tf.nn.softplus(self.ron)

        phit = tf.math.mod(elapsed - s, tau) / tau
        kt = tf.where(
            tf.less(phit, 0.5 * ron),
            2 * phit * ron,
            tf.where(tf.less(phit, ron), 2.0 - 2 * phit / ron, alpha * phit),
            )

        z = (
                tf.matmul(inputs, self.input_kernel)
                + tf.matmul(hidden_state, self.recurrent_kernel)
                + self.bias
        )
        i, ig, fg, og = tf.split(z, 4, axis=-1)

        input_activation = tf.nn.tanh(i)
        input_gate = tf.nn.sigmoid(ig)
        forget_gate = tf.nn.sigmoid(fg + 1.0)
        output_gate = tf.nn.sigmoid(og)

        c_tilde = cell_state * forget_gate + input_activation * input_gate
        c = kt * c_tilde + (1.0 - kt) * cell_state

        h_tilde = tf.nn.tanh(c_tilde) * output_gate
        h = kt * h_tilde + (1.0 - kt) * hidden_state

        return h, [c, h]