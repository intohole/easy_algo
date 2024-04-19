from easy_algo.util.constant import TaskType


class Model:

    def __init__(self,task_type = TaskType.CLASSIFICATION):
        self.task_type = task_type

    def train(self, x_train, y_train, *args, **kwargs):
        raise NotImplementedError

    def predict(self, x_test, *args, **kwargs):
        raise NotImplementedError

