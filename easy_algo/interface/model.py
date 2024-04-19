class BaseModel:
    def __init__(self):
        self.model = None

    def build(self):
        pass

    def train(self, x, y):
        pass

    def predict(self, x):
        pass

    def evaluate(self, x, y):
        pass