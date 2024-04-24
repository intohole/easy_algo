class BaseModel:
    def __init__(self, *args, **kwargs):
        self._model = None
        self.params = None
        self.group = None

    @property
    def model(self):
        if self._model is None:
            self.build()
        return self._model

    def build(self):
        pass

    def train(self, x, y):
        pass

    def predict(self, x):
        pass

    def evaluate(self, x, y):
        pass
