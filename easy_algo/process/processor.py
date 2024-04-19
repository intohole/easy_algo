from easy_algo.operation import *
from sklearn.model_selection import train_test_split
from random import randint
from easy_algo.util.collection_utils import *


class Preprocess:

    def __init__(self, data_source, feature_columns):
        self.data = data_source
        self.feature_columns = feature_columns
        self._build_feature()
        self.group_map = features_to_group_map(feature_columns,"group")

    def group_list_feature(self,group):
        if group not in self.group_map:
            return ""
        return ",".join([feature.feature_name for feature in self.group_map[group]])

    def _build_feature(self):
        # 构建一个feature按照group的map，group对应深度学习的层，这样就可以定制化服务
        # construct feature
        for feature in self.feature_columns:
            if feature.feature_process is None or len(feature.feature_process) == 0:
                continue
            for process in feature.feature_process:
                feature.processors.append(FeatureOperationFactory.create_operation(process, feature))

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


class PandasPreprocess(Preprocess):

    def __init__(self, data, feature_columns, test_size=0.2,random_state= randint(0,100)):
        super().__init__(data, feature_columns)
        self.before_process()
        self.process()
        self.train()
        self.test_size = test_size
        self.random_state = random_state

    def get_columns(self):
        return self.data.columns

    def process(self):
        for feature in self.feature_columns:
            for process in feature.processors:
                process.process(self.data, feature.feature_name)

    def getTrainData(self):
        X, y = self.data.getXy(features=self.feature_columns)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        return x_train, y_train, x_test, y_test

    def train(self):
        pass

    def evaluate(self):
        pass


