# Copyright 2021 The ODE-LSTM Authors. All Rights Reserved.


class HawkLSTMCell(tf.keras.layers.Layer):
    # https://papers.nips.cc/paper/7252-the-neural-hawkes-process-a-neurally-self-modulating-multivariate-point-process.pdf
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = (units, units, units)  # state is a tripple
        self.initializer = "glorot_uniform"
        self.recurrent_initializer = "orthogonal"
        super(HawkLSTMCell, self).__init__(**kwargs)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return (
            tf.zeros([batch_size, self.units], dtype=tf.float32),
            tf.zeros([batch_size, self.units], dtype=tf.float32),
            tf.zeros([batch_size, self.units], dtype=tf.float32),
        )

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if isinstance(input_shape[0], tuple):
            # Nested tuple
            input_dim = input_shape[0][-1]
        self.input_kernel = self.add_weight(
            shape=(input_dim, 7 * self.units),
            initializer=self.initializer,
            name="input_kernel",
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, 7 * self.units),
            initializer=self.recurrent_initializer,
            name="recurrent_kernel",
        )
        self.bias = self.add_weight(
            shape=(7 * self.units),
            initializer=tf.keras.initializers.Zeros(),
            name="bias",
        )

        self.built = True

    def call(self, inputs, states):
        c, c_bar, h = states
        k = inputs[0]  # Is the input
        delta_t = inputs[1]  # is the elapsed time

        z = (
                tf.matmul(k, self.input_kernel)
                + tf.matmul(h, self.recurrent_kernel)
                + self.bias
        )
        i, ig, fg, og, ig_bar, fg_bar, d = tf.split(z, 7, axis=-1)

        input_activation = tf.nn.tanh(i)
        input_gate = tf.nn.sigmoid(ig)
        input_gate_bar = tf.nn.sigmoid(ig_bar)
        forget_gate = tf.nn.sigmoid(fg)
        forget_gate_bar = tf.nn.sigmoid(fg_bar)
        output_gate = tf.nn.sigmoid(og)
        delta_gate = tf.nn.softplus(d)

        new_c = c * forget_gate + input_activation * input_gate
        new_c_bar = c_bar * forget_gate_bar + input_activation * input_gate_bar

        c_t = new_c_bar + (new_c - new_c_bar) * tf.exp(-delta_gate * delta_t)
        output_state = tf.nn.tanh(c_t) * output_gate

        return output_state, [new_c, new_c_bar, output_state]