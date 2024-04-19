from easy_algo.source.source import PandaDataSource
from easy_algo.operation import *


class Preprocess:

    def __init__(self, data_source, feature_columns):
        self.data = data_source
        self.feature_columns = feature_columns
        self._build_feature()

    def _build_feature(self):
        # construct feature
        for feature in self.feature_columns:
            if feature.feature_process is None or len(feature.feature_process) == 0:
                continue
            for process in feature.feature_process:
                feature.processors.append(FeatureOperationFactory.create_operation(process, feature))

    def before_process(self):
        pass

    def after_process(self):
        pass

    def process(self):
        pass

    def get_columns(self):
        pass


class PandasPreprocess(Preprocess):

    def __init__(self, data,feature_columns):
        super().__init__(data, feature_columns)
        self.before_process()
        self.process()
        self.after_process()

    def get_columns(self):
        return self.data.columns

    def process(self):
        for feature in self.feature_columns:
            for process in feature.processors:
                process.process(self.data, feature.feature_name)



