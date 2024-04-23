import tensorflow as tf
from tensorflow.keras.layers import Layer


class LabelSmoothing(Layer):
    """
    Keras层，用于实现标签平滑。
    """

    def __init__(self, smoothing=0.1, **kwargs):
        super(LabelSmoothing, self).__init__(**kwargs)
        self.confidence = None
        self.smoothing = smoothing

    def build(self, input_shape):
        # 为层创建一个可训练的权重
        self.confidence = self.add_weight(
            name='confidence',
            shape=(input_shape[-1],),
            initializer='ones',
            trainable=False
        )
        super(LabelSmoothing, self).build(input_shape)

    def call(self, inputs, training=None):
        """
        实现标签平滑的逻辑。
        """
        assert len(inputs) == 2  # 确保输入有两个：标签和模型输出
        labels, predictions = inputs

        # 确保标签平滑仅在训练时应用
        if training:
            labels = (1 - self.smoothing) * labels + self.smoothing / tf.cast(tf.shape(labels)[-1], tf.float32)
        return labels

    def get_config(self):
        # 让层可序列化
        config = super(LabelSmoothing, self).get_config()
        config.update({'smoothing': self.smoothing})
        return config
