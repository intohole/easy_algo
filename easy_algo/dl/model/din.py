import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Dense, Dropout,concatenate, Input, Lambda
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


# DINLayer类用于实现注意力机制的DIN层
class DINLayer(Layer):
    def __init__(self, embedding_size, dense_size, attention_size, dropout_rate, **kwargs):
        # 初始化父类
        super(DINLayer, self).__init__(**kwargs)

        # 设置DIN层的参数
        self.embedding_size = embedding_size
        self.dense_size = dense_size
        self.attention_size = attention_size
        self.dropout_rate = dropout_rate

        # 创建用户和物品的嵌入层
        self.user_embedding = Embedding(input_dim=user_feature_size, output_dim=embedding_size)
        self.item_embedding = Embedding(input_dim=item_feature_size, output_dim=embedding_size)

        # 创建用户和物品的偏置层
        self.user_bias = Dense(dense_size)
        self.item_bias = Dense(dense_size)

        # 创建注意力层
        self.attention_layer = Dense(attention_size, activation='tanh')

        # 创建注意力权重层
        self.attention_weights = Dense(1, activation='softmax')

        # 创建合并层
        self.merge_layer = Dense(dense_size, activation='sigmoid')

        # 创建dropout层
        self.dropout_layer = Dropout(dropout_rate)

    def build(self, input_shape):
        # 构建层
        self.built = True

    def call(self, inputs):
        # 获取输入
        user_embedding, item_embedding, user_bias, item_bias = inputs

        # 获取用户和物品的嵌入表示
        user_embedding = self.user_embedding(user_embedding)
        item_embedding = self.item_embedding(item_embedding)

        # 计算点积
        dot_product = K.dot(user_embedding, K.transpose(item_embedding))

        # 应用用户和物品的偏置
        dot_product = dot_product * user_bias
        dot_product = dot_product * item_bias

        # 应用物品的嵌入表示
        dot_product = dot_product * item_embedding

        # 计算注意力输出
        attention_output = self.attention_layer(dot_product)

        # 计算注意力权重
        attention_weights = self.attention_weights(attention_output)

        # 应用dropout
        attention_weights = self.dropout_layer(attention_weights)

        # 归一化注意力权重
        normalized_attention_weights = attention_weights / K.l2_normalize(attention_weights, axis=-1)

        # 应用注意力权重调整点积
        attention_output = dot_product * normalized_attention_weights

        # 应用合并层
        merged_output = self.merge_layer(attention_output)

        return merged_output

    def compute_output_shape(self, input_shape):
        # 计算输出形状
        return (input_shape[0], self.dense_size)


# DINModel类用于实现完整的DIN模型
class DINModel(Model):
    def __init__(self, embedding_size, dense_size, attention_size, dropout_rate, **kwargs):
        # 初始化父类
        super(DINModel, self).__init__(**kwargs)

        # 设置DIN模型的参数
        self.embedding_size = embedding_size
        self.dense_size = dense_size
        self.attention_size = attention_size
        self.dropout_rate = dropout_rate

        # 创建用户和物品的嵌入层
        self.user_embedding = Embedding(input_dim=user_feature_size, output_dim=embedding_size)
        self.item_embedding = Embedding(input_dim=item_feature_size, output_dim=embedding_size)

        # 创建用户和物品的偏置层
        self.user_bias = Dense(dense_size)
        self.item_bias = Dense(dense_size)

        # 创建DIN层
        self.din_layer = DINLayer(embedding_size, dense_size, attention_size, dropout_rate)

        # 创建合并层
        self.merge_layer = Dense(dense_size, activation='sigmoid')

        # 创建dropout层
        self.dropout_layer = Dropout(dropout_rate)

        # 创建输出层
        self.output_layer = Dense(1, activation='sigmoid')

    def call(self, inputs):
        # 获取输入
        user_embedding, item_embedding, user_bias, item_bias = inputs

        # 获取用户和物品的嵌入表示
        user_embedding = self.user_embedding(user_embedding)
        item_embedding = self.item_embedding(item_embedding)

        # 获取用户和物品的偏置表示
        user_bias = self.user_bias(user_bias)
        item_bias = self.item_bias(item_bias)

        # 应用DIN层
        attention_output = self.din_layer([user_embedding, item_embedding, user_bias, item_bias])
        attention_output = self.merge_layer(attention_output)
        attention_output = self.dropout_layer(attention_output)

        # 应用输出层
        output = self.output_layer(attention_output)

        return output

    def compute_output_shape(self, input_shape):
        # 计算输出形状
        return input_shape[0], 1
