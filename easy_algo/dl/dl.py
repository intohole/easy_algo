from easy_algo.dl.model_builder import ModelBuilder
from easy_algo.interface.model import BaseModel
from easy_algo.dl.train_warp import *
from easy_algo.util.constant import Optimizer, Loss


class DL(BaseModel):

    def __init__(self, model_config, schema, trainer, epochs=100, optimizer=Optimizer.ADAM,
                 loss=Loss.MEAN_SQUARED_ERROR,
                 user_define_loss=None, metrics=None, callbacks=None, **kwargs):
        super().__init__()
        self.trainer = trainer
        self.user_define_loss = user_define_loss
        self.model_config = model_config
        self.model_builder = None
        self.loss = loss
        self.metrics = metrics
        self.callbacks = callbacks
        self.optimizer = optimizer
        self.schema = schema
        self.epochs = epochs
        if trainer is not None:
            if trainer == 'binaryAcc':
                self.model_compile = BinaryAcc()
            elif trainer == 'multiAcc':
                self.model_compile = MultiClass()
            elif trainer == 'regressionMae':
                self.model_compile = RegressionMae()
            else:
                raise NotImplementedError("Trainer is not supported")

        if isinstance(model_config, str):
            raise NotImplementedError
        elif isinstance(model_config, list):
            self.model_builder = ModelBuilder(model_config, schema=self.schema)
        else:
            raise TypeError('model_config is not of type str or list')
        self._model = None

    def build(self):
        self._model = self.model_builder.model
        self.model_compile.compile(model=self._model)

    def fit(self, x, y):
        self.model.fit(x, y, epochs=self.epochs)

    def predict(self, x):
        self.model.predict(x)

    def evaluate(self, x, y):
        pass
