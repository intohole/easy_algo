from easy_algo.feature.feature import *


class FeatureSchema:

    def __init__(self, category_cols=None, features=None, data_source=None):
        if features is None and data_source is None:
            raise ValueError("FeatureSchema needs features or data_source")
        self.columns = data_source.columns if data_source is not None else []
        self.category_cols = category_cols
        self.features = {}
        self.build_features()

    def build_features(self):
        for col in self.columns:
            if col in self.category_cols:
                feature = CategoryFeature(feature_name=col, voca_size=100, out_dim=10)
            else:
                feature = DenseFeature(feature_name=col)
            self.features[col] = feature

    def update_features(self, col_names, attr, value):
        # 更新特定列的Feature对象的属性
        for col in col_names:
            if col in self.features:
                setattr(self.features[col], attr, value)
                continue
            raise ValueError(f"Column {col} not found in features.")

    def __getitem__(self, item):
        if isinstance(item, str) and item in self.features:
            return self.features[item]
        super(FeatureSchema, self).__getitem__(item)
