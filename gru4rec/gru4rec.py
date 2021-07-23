import tensorflow as tf


class Gru4Rec(tf.keras.Model):
    def __init__(self, input_dim, rnn_units=100):
        super().__init__(self)
        self.gru = tf.keras.layers.GRU(rnn_units, return_state=True, dropout=0.2)
        self.dense1 = tf.keras.layers.Dense(input_dim, activation="tanh")
        self.dense2 = tf.keras.layers.Dense(input_dim, activation=None)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense1(x, training=training)
        x = self.dense2(x, training=training)

        if return_state:
            return x, states
        return x
