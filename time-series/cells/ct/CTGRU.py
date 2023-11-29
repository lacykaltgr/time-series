# Copyright 2021 The ODE-LSTM Authors. All Rights Reserved.


class CTGRU(tf.keras.layers.Layer):
    # https://arxiv.org/abs/1710.04110
    def __init__(self, units, M=8, **kwargs):
        self.units = units
        self.M = M
        self.state_size = units * self.M

        # Pre-computed tau table (as recommended in paper)
        self.ln_tau_table = np.empty(self.M)
        self.tau_table = np.empty(self.M)
        tau = 1.0
        for i in range(self.M):
            self.ln_tau_table[i] = np.log(tau)
            self.tau_table[i] = tau
            tau = tau * (10.0 ** 0.5)

        super(CTGRU, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if isinstance(input_shape[0], tuple):
            # Nested tuple
            input_dim = input_shape[0][-1]

        self.retrieval_layer = tf.keras.layers.Dense(
            self.units * self.M, activation=None
        )
        self.detect_layer = tf.keras.layers.Dense(self.units, activation="tanh")
        self.update_layer = tf.keras.layers.Dense(self.units * self.M, activation=None)
        self.built = True

    def call(self, inputs, states):
        elapsed = 1.0
        if (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) > 1:
            elapsed = inputs[1]
            inputs = inputs[0]

        batch_dim = tf.shape(inputs)[0]

        # States is actually 2D
        h_hat = tf.reshape(states[0], [batch_dim, self.units, self.M])
        h = tf.reduce_sum(h_hat, axis=2)
        states = None  # Set state to None, to avoid misuses (bugs) in the code below

        # Retrieval
        fused_input = tf.concat([inputs, h], axis=-1)
        ln_tau_r = self.retrieval_layer(fused_input)
        ln_tau_r = tf.reshape(ln_tau_r, shape=[batch_dim, self.units, self.M])
        sf_input_r = -tf.square(ln_tau_r - self.ln_tau_table)
        rki = tf.nn.softmax(logits=sf_input_r, axis=2)

        q_input = tf.reduce_sum(rki * h_hat, axis=2)
        reset_value = tf.concat([inputs, q_input], axis=1)
        qk = self.detect_layer(reset_value)
        qk = tf.reshape(qk, [batch_dim, self.units, 1])  # in order to broadcast

        ln_tau_s = self.update_layer(fused_input)
        ln_tau_s = tf.reshape(ln_tau_s, shape=[batch_dim, self.units, self.M])
        sf_input_s = -tf.square(ln_tau_s - self.ln_tau_table)
        ski = tf.nn.softmax(logits=sf_input_s, axis=2)

        # Now the elapsed time enters the state update
        base_term = (1 - ski) * h_hat + ski * qk
        exp_term = tf.exp(-elapsed / self.tau_table)
        exp_term = tf.reshape(exp_term, [batch_dim, 1, self.M])
        h_hat_next = base_term * exp_term

        # Compute new state
        h_next = tf.reduce_sum(h_hat_next, axis=2)
        h_hat_next_flat = tf.reshape(h_hat_next, shape=[batch_dim, self.units * self.M])
        return h_next, [h_hat_next_flat]