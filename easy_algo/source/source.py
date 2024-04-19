import pandas as pd


class DataSource(object):

    def __init__(self, data):
        self.data = None


    def select(self,condition):
        raise NotImplementedError

    def filter(self,condition):
        raise NotImplementedError

    def update(self,condition,value):
        raise NotImplementedError

    def drop(self,condition):
        raise NotImplementedError



class PandaDataSource(DataSource):

    def __init__(self, data_frame, data_path, data_sep='\0'):

        if data_frame is None:
            if data_path is None:
                raise ValueError("Path cannot be None")
            self.data_frame = pd.read_csv(data_path, data_sep=data_sep)
        else:
            self.data_frame = data_frame
        super().__init__(self.data_frame)

    def select(self, condition):
        if condition is None:
            raise ValueError("Condition cannot be None")
        return self.data_frame[condition]

    def filter(self, condition):
        if condition is None:
            raise ValueError("Condition cannot be None")
        self.data_frame = self.data_frame[condition]
        return self.data_frame

    def update(self, condition, value):
        if condition is None:
            raise ValueError("Condition cannot be None")
        self.data_frame[condition] = value


