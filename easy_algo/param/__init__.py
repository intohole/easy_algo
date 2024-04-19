class Param(object):

    def __init__(self, task_type):
        self.task_type = task_type

    def get_config(self):
        raise NotImplementedError

