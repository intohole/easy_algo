import pandas as pd
from easy_algo.feature.feature import FEATURE_TYPE

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

    def get_xy(self, features):
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

    def get_xy(self, features, test_condition=None):
        if features is None:
            raise ValueError("Features cannot be None")
        feature_names = [_.feature_name for _ in features if _.train_able and _.type == FEATURE_TYPE.feature]
        label_names = [_.feature_name for _ in features if _.train_able and _.type == FEATURE_TYPE.label]
        return self.data_frame[feature_names].values, self.data_frame[label_names].values

    def get_xy(self, features, test_condition=None):
        if features is None:
            raise ValueError("Features cannot be None")

        # 按照group字段划分特征和标签
        groups = defaultdict(list)
        for feature in features:
            if feature.train_able and feature.type == FeatureType.feature:
                groups[feature.group].append(feature)

        # 获取每个group的字段名称
        group_feature_names = {}
        for group, features in groups.items():
            feature_names = [feature.feature_name for feature in features]
            label_names = [feature.feature_name for feature in features if feature.type == FeatureType.label]
            group_feature_names[group] = (feature_names, label_names)

        # 根据test_condition进行划分
        if test_condition is not None:
            # 这里需要实现具体的划分逻辑，例如使用DataFrame的query方法
            pass
        else:
            # 如果没有test_condition，则使用原始的切分规则
            pass

        # 根据group和划分结果获取特征和标签
        X = {}
        y = {}
        for group, (feature_names, label_names) in group_feature_names.items():
            X[group] = self.data_frame[feature_names].values
            y[group] = self.data_frame[label_names].values

        return X, y
