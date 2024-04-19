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
