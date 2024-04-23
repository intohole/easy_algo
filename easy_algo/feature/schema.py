from easy_algo.feature.feature import *


class FeatureSchema:
    def __init__(self, category_cols=None, dtypes=None, features=None, data_source=None):
        self.dtypes = dtypes
        if features is None and data_source is None:
            raise ValueError("FeatureSchema needs features or data_source")
        self.columns = data_source.columns if data_source is not None else []
        self.category_cols = category_cols
        self.features = features
        self.build_features()

    def build_features(self):
        for col in self.columns:
            if col in self.category_cols:
                feature = self.create_category_feature(col)
            else:
                feature = self.create_dense_feature(col)
            self.features[col] = feature

    def create_category_feature(self, col):
        """创建一个类别特征。"""
        return CategoryFeature(col_name=col)

    def create_dense_feature(self, col):
        """创建一个密集特征。"""
        return DenseFeature(col_name=col)

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

    @staticmethod
    def build_feature_schema(category_cols, columns, dtypes):
        if category_cols is None:
            category_cols = set()
        if columns is None or dtypes is None:
            raise ValueError("FeatureSchema needs columns or dtypes")
        if len(columns) != len(category_cols):
            raise ValueError("FeatureSchema needs columns and category_cols")
        features = []
        for col, dtype in zip(columns, dtypes):
            if col in category_cols:
                feature = CategoryFeature(col_name=col)
            else:
                feature = DenseFeature(col_name=col)
            feature.shape = 1
            feature._origin_type = dtype
            features.append(feature)
        schema = FeatureSchema(features=features)
        schema.dtypes = dtypes
        schema.columns = columns
        return schema
