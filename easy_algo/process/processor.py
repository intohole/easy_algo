from easy_algo.operation import *
from sklearn.model_selection import train_test_split
from random import randint
from easy_algo.util.collection_utils import *
from easy_algo.feature.feature import FeatureGroup
from easy_algo.util.manager import ModelFactory


class Preprocess:

    def __init__(self, data_source, feature_columns):
        self.groups = None
        self.data = data_source
        self.feature_columns = feature_columns
        self._build_feature()
        self.build_groups()  #
        self.group_map = features_to_group_map(self.feature_columns, "group")

    def build_groups(self):
        self.groups = [FeatureGroup(k, v) for k, v in self.group_map.items()]

    def group_list_feature(self, group):
        if group not in self.groups:
            return ""
        return ",".join([feature.feature_name for feature in self.group_map[group]])

    def _build_feature(self):
        # 构建一个feature按照group的map，group对应深度学习的层，这样就可以定制化服务
        # construct feature
        for feature in self.feature_columns:
            if feature.feature_process is None or len(feature.feature_process) == 0:
                continue
            for process in feature.feature_process:
                if isinstance(process, str):
                    feature.processors.append(FeatureOperationFactory.create_operation(process, feature))
                else:
                    feature.processors.append(process)

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

    def __init__(self, data, feature_columns, model_config, test_size=0.2, random_state=randint(0, 100), *args,
                 **kwargs):
        super().__init__(data, feature_columns)
        self.before_process()
        self.process()
        self.train()
        self.test_size = test_size
        self.random_state = random_state
        self.model_config = model_config
        self.model = self.build_model(*args, **kwargs)
        self.x_train, self.y_train, self.x_test, self.y_test = self.split_data()

    def build_model(self, *args, **kwargs):
        if isinstance(self.model_config, str):
            return ModelFactory.create(self.model_config, *args, **kwargs)

    def get_columns(self):
        return self.data.columns

    def process(self):
        for feature in self.feature_columns:
            for process in feature.processors:
                process.process(self.data, feature.feature_name)

    def split_data(self):
        X, y = self.data.getXy(features=self.feature_columns)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size,
                                                            random_state=self.random_state)
        return x_train, y_train, x_test, y_test

    def train(self):
        self.model.build()
        self.model.fit(self.getTrainData())

    def evaluate(self):
        self.model.evaluate(self.getTrainData())

