from easy_algo.dl.layer.dnn import *
import tensorflow as tf


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
        :param inputs: 输入特征，形状为 (batch_size, input_dim)
        :return: 所有任务的输出
        """
        # 应用专家网络
        # 每个专家网络处理输入特征并生成一个输出
        # experts_outputs 的形状: [(batch_size, expert_output_dim), ...] (num_experts 个元素)
        experts_outputs = [expert(inputs) for expert in self.experts]

        # 将所有专家网络的输出堆叠成一个张量
        # experts_outputs 的形状: (batch_size, expert_output_dim, num_experts)
        experts_outputs = tf.stack(experts_outputs, axis=-1)

        # 应用门控机制
        # 每个任务有自己的门控网络，决定如何组合专家网络的输出
        # gates_outputs 的形状: [(batch_size, num_experts), ...] (num_tasks 个元素)
        gates_outputs = [gate(inputs) for gate in self.gates]

        # 将所有门控网络的输出堆叠成一个张量
        # gates_outputs 的形状: (batch_size, num_experts, num_tasks)
        gates_outputs = tf.stack(gates_outputs, axis=-1)

        # 计算加权输出
        # 将专家网络的输出与门控网络的输出相乘
        # 利用广播机制，结果形状: (batch_size, expert_output_dim, num_tasks)
        weighted_outputs = experts_outputs * gates_outputs

        # 汇总所有任务的输出
        # 在专家网络维度上求和，为每个任务生成一个输出
        # task_outputs 的形状: (batch_size, expert_output_dim, num_tasks)
        task_outputs = tf.reduce_sum(weighted_outputs, axis=-1)

        return task_outputs

