from easy_algo.util.constant import Optimizer, Loss, Metrics

from easy_algo.util.constant import Optimizer, Loss, Metrics


class ModelTrainner:
    def __init__(self):
        pass

    def compile(self, *args, **kwargs):
        raise NotImplementedError

    def _compile(self, model, optimizer=Optimizer.ADAM, loss=None, metrics=None, **kwargs):
        if metrics is None:
            metrics = [Metrics.ACC.value]
        optimizer_name = optimizer.value
        if loss is not None:
            _loss = loss.value
        else:
            raise ValueError("Loss function must be specified")

        model.compile(optimizer=optimizer_name, loss=_loss, metrics=metrics, **kwargs)
        return model


class BinaryAcc(ModelTrainner):
    def __init__(self):
        super(BinaryAcc, self).__init__()

    def compile(self, model, *args, **kwargs):
        return self._compile(model, optimizer=Optimizer.SGD, loss=Loss.BINARY_CROSSENTROY,
                             metrics=[Metrics.ACC.value], **kwargs)


class MultiClass(ModelTrainner):
    def __init__(self):
        super(MultiClass, self).__init__()

    def compile(self, model, *args, **kwargs):
        return self._compile(model, optimizer=Optimizer.SGD, loss=Loss.CATEGORICAL_CROSS_ENTROPY)


class RegressionMae(ModelTrainner):
    def __init__(self):
        super(RegressionMae, self).__init__()

    def compile(self, model, *args, **kwargs):
        return self._compile(model, optimizer=Optimizer.SGD, loss=Loss.MEAN_SQUARED_ERROR,
                             metrics=[Metrics.MAE.value, Metrics.MSE.value])
