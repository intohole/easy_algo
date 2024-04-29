class DataSource(object):

    def __init__(self, data):
        self.data = None

    def select(self, condition):
        raise NotImplementedError

    def filter(self, condition):
        raise NotImplementedError

    def update(self, condition, value):
        raise NotImplementedError

    def drop(self, condition):
        raise NotImplementedError

    def dtypes(self):
        pass
