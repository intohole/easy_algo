from enum import Enum


class Optimizer(Enum):
    ADAM = 'adam'
    SGD = 'sgd'
    RMSPROP = 'rmsprop'
    ADAGRAD = 'adagrad'


class Loss(Enum):
    MEAN_SQUARED_ERROR = 'mean_squared_error'
    SIGMOID_CROSS_ENTROPY = 'sigmoid_cross_entropy'
    SOFTMAX_CROSS_ENTROPY = 'softmax_cross_entropy'
    CATEGORICAL_CROSS_ENTROPY = 'categorical_cross_entropy'
    HINGE = 'hinge'
    MAE = 'mean_absolute_error'


class Metrics(Enum):
    ACC = 'accuracy'
    SQUARED_ERROR = 'mean_squared_error'
    CATEGORICAL_ACCURACY = 'categorical_accuracy'
    TOP_K_CATEGORICAL_ACCURACY = 'top_k_categorical_accuracy'
    AUC = 'auc'
    SENSITIVITY = 'sensitivity'
    SPECIFICITY = 'specificity'
    POSITIVE_PREDICTIVE_VALUE = 'positive_predictive_value'
    NEGATIVE_PREDICTIVE_VALUE = 'negative_predictive_value'
    PRECISION = 'precision'
    RECALL = 'recall'
    F1_SCORE = 'f1_score'


class Callback(Enum):
    EARLY_STOPPING = 'early_stopping'
    MODEL_CHECKPOINT = 'model_checkpoint'
    REDUCE_LR_ON_PLATEAU = 'reduce_lr_on_plateau'


def compile_model(model, optimizer=Optimizer.ADAM, loss=Loss.MEAN_SQUARED_ERROR, metrics=None, callbacks=None):
    """
    编译Keras模型的函数。

    参数:
    - model: 一个未编译的Keras模型。
    - optimizer: 优化器枚举类型。
    - loss: 损失函数枚举类型。
    - metrics: 评估指标列表。
    - callbacks: 回调函数枚举类型列表。

    返回:
    - 一个编译好的Keras模型。
    """
    if metrics is None:
        metrics = [Metrics.ACC]
    optimizer_name = optimizer.value
    loss_name = loss.value

    if callbacks is None:
        callbacks = []

    model.compile(optimizer=optimizer_name, loss=loss_name, metrics=metrics, callbacks=callbacks)
    return model
