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
        self.dense_array = [Dense(units=unit, activation=_activation) for unit, _activation in
                            zip(hidden_units, self._activation)]

    def call(self, inputs):
        output = self.denses[0](inputs)
        for dense in self.denses[1:]:
            output = dense(output)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape, self.hidden_units[-1]


class MMoELayer(tf.keras.layers.Layer):
    def __init__(self, num_experts, num_tasks, expert_hidden_units=None, gate_activation='softmax', **kwargs):
        """
        MMoE层构造函数。
        :param num_experts: 专家网络的数量
        :param num_tasks: 任务的数量
        :param expert_hidden_units: 每个专家网络的隐藏层单元数
        """
        super(MMoELayer, self).__init__(**kwargs)
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.experts = [DeepLayer(hidden_units=expert_hidden_units, activation='relu') for _ in range(num_experts)]
        self.gates = [Dense(units=num_experts, activation=gate_activation) for _ in range(num_tasks)]

    def call(self, inputs):
        """
        MMoE层的计算方法。
        :param inputs: 输入特征
        :return: 所有任务的输出
        """
        # 应用专家网络
        experts_outputs = [expert(inputs) for expert in self.experts]
        experts_outputs = tf.stack(experts_outputs, axis=-1)

        # 应用门控机制
        gates_outputs = [gate(inputs) for gate in self.gates]
        gates_outputs = tf.stack(gates_outputs, axis=-1)
        weighted_outputs = experts_outputs * gates_outputs

        # 汇总所有任务的输出
        task_outputs = tf.reduce_sum(weighted_outputs, axis=-1)
        return task_outputs
