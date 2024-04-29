from easy_algo.struct.listener import *


class Processor(ListenerInterface):

    def __init__(self, feature_columns):
        self.groups = None
        self.data = None
        self.feature_columns = feature_columns

    def before_process(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def process(self):
        raise NotImplementedError

    def get_columns(self):
        raise NotImplementedError

    def getTrainData(self):
        raise NotImplementedError