class Model:

    def __init__(self):
        pass

    def train(self, x_train, y_train, *args, **kwargs):
        raise NotImplementedError

    def predict(self, x_test, *args, **kwargs):
        raise NotImplementedError

    def predict_proba(self, x_test, *args, **kwargs):
        raise NotImplementedError
