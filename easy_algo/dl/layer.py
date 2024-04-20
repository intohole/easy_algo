import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten
from tensorflow.keras.models import Model


# Wide部分
class WideLayer(tf.keras.layers.Layer):
    def __init__(self, voca_size, output_dim, **kwargs):
        super(WideLayer, self).__init__()
        self.output_dim = output_dim
        self.voca_size = voca_size
        self.embedding = Embedding(input_dim=voca_size + 1, output_dim=self.output_dim, input_length=1)
        self.dense = Dense(units=self.output_dim)

    def call(self, inputs):
        wide_features = self.embedding(inputs)
        flattened = Flatten()(wide_features)
        return self.dense(flattened)


# Deep部分
class DeepLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_units, activation='relu', **kwargs):
        super(DeepLayer, self).__init__()
        self.hidden_units = hidden_units
        self.activation = activation
        self.denses = [Dense(units=_, activation=activation) for _ in range(hidden_units)]

    def call(self, inputs):
        output = self.denses[0](inputs)
        for dense in self.denses[1:]:
            output = dense(output)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape, self.hidden_units[-1]


