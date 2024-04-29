from easy_algo.interface.source import DataSource
import pandas as pd


class PandaDataSource(DataSource):

    def __init__(self, data_frame, data_path, data_sep='\0'):
        if data_frame is None and data_path is None:
            raise ValueError("Either data_frame or data_path must be provided")
        if data_frame is None:
            self.data_frame = pd.read_csv(data_path, sep=data_sep)
        else:
            self.data_frame = data_frame
        super().__init__(self.data_frame)

    @property
    def columns(self):
        return self.data_frame.columns

    @property
    def dtypes(self):
        return self.data_frame.dtypes

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

    def __getitem__(self, item):
        return self.data_frame[item]


