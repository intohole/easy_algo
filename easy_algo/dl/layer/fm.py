from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K


class FMLayer(Layer):
    def __init__(self, input_dim=None, factor_order=10, **kwargs):
        if input_dim is None:
            raise ValueError("input_dim cannot be None")
        self.kernel = None
        self.input_dim = input_dim  # 输入维度，即特征向量的长度
        self.factor_order = factor_order  # 因子阶数，控制交叉项的复杂度
        super(FMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # 创建交叉层的权重
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.input_dim, self.factor_order),
                                      initializer='glorot_uniform',
                                      trainable=True)
        super(FMLayer, self).build(input_shape)

    def call(self, X):
        # 计算交叉项
        # 这里使用了Keras的后端张量操作库K来计算
        # K.dot(X, self.kernel) 计算输入X和权重kernel的点积
        square_of_sum = K.pow(K.dot(X, self.kernel), 2)
        # K.pow(X, 2) 对输入X的每个元素进行平方
        # K.pow(self.kernel, 2) 对权重kernel的每个元素进行平方
        # K.dot(K.pow(X, 2), K.pow(self.kernel, 2)) 计算平方后的X和平方后的kernel的点积
        sum_of_square = K.dot(K.pow(X, 2), K.pow(self.kernel, 2))
        # 返回交叉项的输出，这里使用0.5是为了简化最后的求和表达式
        return 0.5 * K.sum(square_of_sum - sum_of_square, axis=1, keepdims=True)

    def compute_output_shape(self, input_shape):
        # 定义输出形状，这里输出的是一个二维张量，第一个维度是输入的样本数量，第二个维度是1
        return input_shape[0], 1
