from easy_algo.source.source import DataSource
from collections import defaultdict
from easy_algo.util.constant import FeatureType
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

        # 按照group字段划分特征和标签
        groups = defaultdict(list)
        for feature in features:
            if feature.train_able and feature.type == FeatureType.Feature:
                groups[feature.group].append(feature)

        # 获取每个group的字段名称
        group_feature_names = {}
        for group, features in groups.items():
            feature_names = [feature.feature_name for feature in features]
            label_names = [feature.feature_name for feature in features if feature.type == FeatureType.label]
            group_feature_names[group] = (feature_names, label_names)

        # 根据group和划分结果获取特征和标签
        X = {}
        y = {}
        for group, (feature_names, label_names) in group_feature_names.items():
            X[group] = self.data_frame[feature_names].values
            y[group] = self.data_frame[label_names].values

        return X, y

    def dtypes(self):
        return self.data_frame.dtypes
