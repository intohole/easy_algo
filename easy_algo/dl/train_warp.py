from easy_algo.util.constant import Optimizer, Loss, Metrics


def compile_model(model, optimizer=Optimizer.ADAM, loss=Loss.MEAN_SQUARED_ERROR,userDefineLoss = None , metrics=None, callbacks=None,**kwargs):
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
        metrics = [Metrics.ACC.value]
    optimizer_name = optimizer.value
    _loss = None
    if userDefineLoss is not None:
        _loss = userDefineLoss
    else:
        _loss = loss.value

    if callbacks is None:
        callbacks = []

    model.compile(optimizer=optimizer_name, loss=_loss, metrics=metrics, callbacks=callbacks,**kwargs)
    return model
