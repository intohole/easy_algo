import tensorflow as tf
from tensorflow.keras.losses import Loss


class LabelSmoothLoss(Loss):
    def __init__(self, label_smoothing=0.1, reduction='auto', name='smooth_crossentropy'):
        super(LabelSmoothLoss, self).__init__(reduction=reduction, name=name)
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        """
        实现带标签平滑的交叉熵损失计算。

        参数:
        y_true: 真实标签，通常为 one-hot 编码的向量。
        y_pred: 模型的预测输出。

        标签平滑的数学原理基于这样的思想：硬标签（即一个类别概率为1，其他为0）可能导致模型过拟合。
        通过将硬标签稍微“平滑”一下，即将硬标签的值稍微减少一点，并将这些减少的值平均分配给其他类别，
        可以减少模型对于硬标签的敏感度，从而提高模型的泛化能力。

        公式上，对于每个类别 i，平滑后的标签 y_i 为：
        y_i = (1 - label_smoothing) * y_true_i + label_smoothing / num_classes

        其中 y_true_i 是硬标签中第 i 个类别的值，num_classes 是类别总数。
        """
        num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)  # 获取类别总数
        y_true = (1 - self.label_smoothing) * y_true + self.label_smoothing / num_classes
        # 应用标签平滑

        # 计算交叉熵损失
        # 交叉熵损失公式为 H(y_true, y_pred) = -sum(y_true * log(y_pred))
        # 在这里，由于 y_true 已经被平滑处理，因此计算的交叉熵将包含平滑效果
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)
