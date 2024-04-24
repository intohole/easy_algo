from easy_algo.feature.feature import *
from easy_algo.feature.feature_group import *

class FeatureSchema:
    def __init__(self, source=None, columns=None, groups=None, dtypes=None, features=None, data_source=None):
        self.dtypes = dtypes
        self.source = source
        if features is None and data_source is None:
            raise ValueError("FeatureSchema needs features or data_source")
        self.columns = data_source.columns if data_source is not None else []
        self.category_cols = columns
        self.features = features
        self.groups = [] if groups is None else groups
        self.group_map = {}

    def update_features(self, feature_names, attr, value):
        # 更新特定列的Feature对象的属性
        for feature_name in feature_names:
            if feature_name in self.features:
                setattr(self.features[feature_name], attr, value)
                continue
            raise ValueError(f"Column {feature_name} not found in features.")

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
        if len(columns) != len(dtypes):
            raise ValueError("FeatureSchema needs columns or dtypes")
        features = []
        is_contain_dense = False
        for col, dtype in zip(columns, dtypes):
            if col in category_cols:
                feature = CategoryFeature(col_name=col)
            else:
                feature = DenseFeature(col_name=col)
                is_contain_dense = True
            feature.shape = 1
            feature._origin_type = dtype
            features.append(feature)
        # 默认设置group信息
        group_index = 0
        groups = [FeatureGroup("group0")] if is_contain_dense else []
        for feature in features:
            if isinstance(feature, DenseFeature):
                feature.group = "group0"
                groups[0].append_feature(feature)
            elif isinstance(feature, CategoryFeature):
                feature.group = "group%s" % group_index
                groups.append(FeatureGroup("group%s" % group_index,[feature]))
                group_index += 1
        schema = FeatureSchema(features=features, groups=groups, dtypes=dtypes)
        schema.dtypes = dtypes
        schema.columns = columns
        return schema
