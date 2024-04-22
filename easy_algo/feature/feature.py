from easy_algo.util.constant import FeatureType


# 基础的Feature类，用于表示特征
class Feature(object):
    def __init__(self, feature_name, value_type=FeatureType.Feature, feature_process=None, group=None,
                 cover_name=False, shape=None):
        if feature_process is None:
            feature_process = []
        self.value_type = value_type  # 特征值的类型
        self.feature_name = feature_name  # 特征的名称
        self.type = value_type  # 特征的类型
        self.feature_process = feature_process  # 特征的处理方法列表
        self.cover_name = cover_name  # 是否覆盖特征名称
        self.processors = []  # 特征处理器列表
        self.train_able = True  # 如果为True，则此特征是可训练的
        self.group = "default" if group is None else group  # 用于标识模型的组别
        self.shape = shape  # 数据大小，用于向量化准备


# 密集特征类，继承自Feature
class DenseFeature(Feature):
    def __init__(self, feature_name, value_type='feature', feature_process=None):
        super(DenseFeature, self).__init__(feature_name, value_type, feature_process)
        self.bucket_num = None  # 桶的数量（用于分桶特征）
        self.condition_map = {}  # 条件映射


# 类别特征类，继承自Feature
class CategoryFeature(Feature):
    def __init__(self, feature_name, voca_size, out_dim, value_type='feature', feature_process=None):
        super(CategoryFeature, self).__init__(feature_name, value_type, feature_process)
        self.voca_size = voca_size  # 词汇表大小
        self.out_dim = out_dim  # 输出维度


class FeatureGroup:

    def __init__(self, group_name, group_features):
        self.features = group_features
        self.group_name = group_name

    def get_feature_shape(self):
        return sum(_.shape for _ in self.features), 1


class FeatureGroup:
    def __init__(self, group_name, group_features):
        self.group_name = group_name
        self.features = group_features
        self.shape = sum(_.shape for _ in self.features)

    @property
    def features(self):
        """
        获取features属性的值。
        """
        return self._features

    @features.setter
    def features(self, value):
        """
        更新features属性时，自动更新shape属性。
        """
        self._features = value
        self.shape = sum(_.shape for _ in self.features)


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
        if isinstance(item,str) and item in self.features:
            return self.features[item]
        super(FeatureSchema, self).__getitem__(item)
