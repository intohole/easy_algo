import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Flatten


# Wide部分
class WideLayer(tf.keras.layers.Layer):
    def __init__(self, voca_size, output_dim, embedding_input_dim=None, embedding_output_dim=None, flatten_output=True,
                 **kwargs):
        super(WideLayer, self).__init__()
        self.output_dim = output_dim
        self.voca_size = voca_size
        self.embedding_input_dim = embedding_input_dim if embedding_input_dim is not None else voca_size + 1
        self.embedding_output_dim = embedding_output_dim if embedding_output_dim is not None else output_dim
        self.flatten_output = flatten_output
        self.embedding = Embedding(input_dim=self.embedding_input_dim, output_dim=self.embedding_output_dim,
                                   input_length=1)
        self.dense = Dense(units=self.output_dim)

    def call(self, inputs):
        wide_features = self.embedding(inputs)
        if self.flatten_output:
            flattened = Flatten()(wide_features)
        else:
            flattened = wide_features
        return self.dense(flattened)


# Deep部分


