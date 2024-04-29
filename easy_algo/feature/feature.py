from markdown.util import Processor

from easy_algo.util.constant import FeatureType, Event
from easy_algo.struct.listener import *
from easy_algo.operation.operation import FeatureOperation
from easy_algo.operation import FeatureOperationFactory


# 基础的Feature类，用于表示特征
class Feature(Subject):
    def __init__(self, col_name, value_type=FeatureType.Feature, feature_process=None, group=None, cover_name=False,
                 default_value=None, shape=None, dtype=None, fun=None):
        super().__init__()
        if feature_process is None:
            feature_process = []
        self.value_type = value_type  # 特征值的类型
        self.col_name = col_name  # 特征的名称
        self.type = value_type  # 特征的类型
        self.cover_name = cover_name  # 是否覆盖特征名称
        self.processors = feature_process  # 特征处理器列表
        self.train_able = True  # 如果为True，则此特征是可训练的
        self._group = None
        self.group = group  # 用于标识模型的组别
        self.shape = shape  # 数据大小，用于向量化准备
        self.default_value = default_value
        self.dtype = dtype
        self.fun = fun
        self._origin_type = None

    @property
    def group(self):
        return self._group

    @group.setter
    def group(self, group):
        old_group = self._group
        self._group = group
        self.notify_listeners(Event.GROUP_CHANGE, self, old_group, group)

    @property
    def processors(self):
        return self._processors

    @processors.setter
    def processors(self, value):
        processors = []
        if isinstance(value, list):
            for processor in value:
                processors.append(self._get_processor(processor))
        else:
            processors.append(self._get_processor(value))
        self._processors = processors

    def _get_processor(self, processor):
        if isinstance(processor, FeatureOperation):
            return processor
        elif isinstance(processor, str):
            return FeatureOperationFactory.create_operation(processor, feature=self)
        else:
            raise TypeError(f"Processor type {type(processor)} not supported")

    def __call__(self, *args, **kwargs):
        if len(args) == 1:
            if isinstance(args[0], str):
                self._processors.append(self._get_processor(args[0]))
                self.notify_listeners(Event.PROCESS, self, self._processors[-1])
        else:
            raise NotImplementedError


class DenseFeature(Feature):
    def __init__(self, col_name, feature_process=None):
        super(DenseFeature, self).__init__(col_name, feature_process)
        self.bucket_num = None  # 桶的数量（用于分桶特征）


class LabelFeature(Feature):
    def __init__(self, col_name, feature_process=None):
        super(LabelFeature, self).__init__(col_name, feature_process)
        self.label_type = None  # 标签的类型，如二分类、多分类等
        self.classes = None  # 所有可能的标签类别
        self.one_hot_dim = None  # 如果使用独热编码，输出的维度


# 类别特征类，继承自Feature
class CategoryFeature(Feature):
    def __init__(self, col_name, feature_process=None):
        super(CategoryFeature, self).__init__(col_name, feature_process)
        self.voca_size = None  # 词汇表大小
        self.out_dim = None  # 输出维度
