import tensorflow as tf
from tensorflow.keras.layers import Dense


class DeepLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_units=None, activation='relu', **kwargs):
        super(DeepLayer, self).__init__()
        if hidden_units is None:
            hidden_units = [128, 64, 32, 16]
        self.hidden_units = hidden_units
        if isinstance(activation, str):
            self._activation = [activation for _ in range(len(hidden_units))]
        elif isinstance(activation, list):
            if len(hidden_units) != len(activation):
                raise ValueError("hidden_units and activation must be same length")
            self._activation = activation
        self.dense_array = [Dense(units=unit, activation=_activation) for unit, _activation in
                            zip(hidden_units, self._activation)]

    def call(self, inputs):
        output = self.denses[0](inputs)
        for dense in self.denses[1:]:
            output = dense(output)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape, self.hidden_units[-1]
