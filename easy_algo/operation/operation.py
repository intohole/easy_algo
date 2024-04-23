from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer
import numpy as np


class FeatureOperation:
    def __init__(self, name, suffix_name=None):
        self.name = name
        self._feature = None
        self._cover_name = False
        self.suffix_name = suffix_name

    @property
    def feature(self):
        return self._feature

    @property
    def cover_name(self):
        return self._cover_name

    @cover_name.setter
    def cover_name(self, value):
        self._cover_name = value

    @feature.setter
    def feature(self, feature):
        self._feature = feature

    def get_feature_name(self, field):
        if self.cover_name:
            return "%s_%s" % (field, self.suffix_name)
        else:
            return field

    def process(self, data_source, field):
        raise NotImplementedError

    def __getstate__(self):
        # 定义在序列化时需要保存的状态
        return self.__dict__

    def __setstate__(self, state):
        # 定义在反序列化时需要设置的状态
        self.__dict__ = state

    def to_java_code(self):
        raise NotImplementedError


class MinMaxNormalization(FeatureOperation):
    def __init__(self):
        super(MinMaxNormalization, self).__init__("minMaxNor", suffix_name="nor")
        self.min_val = None
        self.max_val = None

    def process(self, data_source, field):
        if self.min_val is None:
            self.min_val = data_source.data[self.name].min()
        if self.max_val is None:
            self.max_val = data_source.data[self.name].max()
        data_source.data[self.get_feature_name(field)] = (data_source.data[field] - self.min_val) / (
                self.max_val - self.min_val)


class Standardization(FeatureOperation):
    def __init__(self):
        super(Standardization, self).__init__("standardNor", suffix_name="std")
        self.std_val = None
        self.mean_val = None

    def process(self, data_source, field):
        if self.mean_val is None:
            self.mean_val = data_source.data[field].mean()
        if self.std_val is None:
            self.std_val = data_source.data[field].std()
        data_source.data[self.get_feature_name(field)] = (data_source.data[field] - self.mean_val) / self.std_val


class FeatureBucket(FeatureOperation):
    def __init__(self, name, method='equal_width', num_bins=5):
        super(FeatureBucket, self).__init__(name, suffix_name='binned')
        self.method = method
        self.num_bins = self._feature.bucket_num
        if self._feature.bucket_num is None:
            raise ValueError("please set feature bucket num")
        self.binning = None
        self._init()

    def _init(self):
        if self.binning is None:
            if self.method == 'equal_width':
                self.binning = KBinsDiscretizer(n_bins=self.num_bins, encode='ordinal', strategy='uniform')
            elif self.method == 'equal_frequency':
                self.binning = KBinsDiscretizer(n_bins=self.num_bins, encode='ordinal', strategy='quantile')
            else:
                raise ValueError("Unsupported binning method")

    def process(self, data_source, field):
        binned_data = self.binning.fit_transform(data_source.data[[field]])
        data_source.data[self.get_feature_name(field)] = binned_data


class FeatureEqualWidth(FeatureBucket):

    def __init__(self):
        super(FeatureEqualWidth, self).__init__(name="equalWidthBucket", method="equal_width", num_bins=num_bins)


class FeatureEqualFrequency(FeatureBucket):

    def __init__(self):
        super(FeatureEqualFrequency).__init__(name="equalFrequencyBucket", method="equal_frequency", num_bins=num_bins)


class OneHotEncoding(FeatureOperation):
    def __init__(self):
        super(OneHotEncoding, self).__init__("oneHot", suffix_name="oh")
        self.encoder = OneHotEncoder()

    def process(self, data_source, field):
        onehot_data = self.encoder.fit_transform(data_source.data[[field]])
        data_source.data[self.get_feature_name(field)] = onehot_data


class LabelEncoding(FeatureOperation):
    def __init__(self):
        super(LabelEncoding, self).__init__("labelEncode", suffix_name="label")
        self.encoder = LabelEncoder()

    def process(self, data_source, field):
        data_source.data[self.get_feature_name(field)] = self.encoder.fit_transform(data_source.data[[field]])


class LogTransformation(FeatureOperation):
    def __init__(self):
        super(LogTransformation, self).__init__("logTransform", suffix_name="log")

    def process(self, data_source, field):
        data_source.data[self.get_feature_name(field)] = np.log(data_source.data[field] + 1)  # 加1以避免对0取对数


class PowerTransformation(FeatureOperation):
    def __init__(self):
        super(PowerTransformation, self).__init__("power", suffix_name="power")
        self.power_transformer = self.power_transform()

    def power_transform(self, data):
        # 使用sklearn的PowerTransformer进行幂转换
        return self.power_transformer.fit_transform(data.reshape(-1, 1))

    def process(self, data_source, field):
        if self.power_transformer is None:
            self.power_transformer = PowerTransformer()
        data_source.data[self.get_feature_name(field)] = self.power_transform(data_source.data[[field]])


class MissingValueImputation(FeatureOperation):
    def __init__(self, name, strategy='mean'):
        super(MissingValueImputation, self).__init__(name, suffix_name="mean")
        self.strategy = strategy
        self.imputer = SimpleImputer(strategy=self.strategy)

    def process(self, data_source, field):
        data_source.data[self.get_feature_name(field)] = self.imputer.fit_transform(data_source.data[[field]])


class MeanMissingValueImputation(MissingValueImputation):
    def __init__(self):
        super(MeanMissingValueImputation, self).__init__("meanImputation", "mean")


class MedianMissingValueImputation(MissingValueImputation):
    def __init__(self):
        super(MedianMissingValueImputation, self).__init__("medianImputation", "median")


class ModeMissingValueImputation(MissingValueImputation):
    def __init__(self):
        super(ModeMissingValueImputation, self).__init__("modeImputation", "most_frequent")


class TimeFeatureHourExtraction(FeatureOperation):
    def __init__(self):
        super(TimeFeatureHourExtraction, self).__init__("hour", suffix_name="hour")

    def process(self, data_source, field):
        # 提取小时
        data_source.data[self.get_feature_name(field)] = data_source.data[field].dt.hour


class TimeFeatureWeekExtraction(FeatureOperation):
    def __init__(self):
        super(TimeFeatureWeekExtraction, self).__init__("week", suffix_name="week")

    def process(self, data_source, field):
        # 提取小时
        data_source.data[self.get_feature_name(field)] = data_source.data[field].dt.dayofweek.astype('int')


class TimeFeatureMonthExtraction(FeatureOperation):
    def __init__(self):
        super(TimeFeatureMonthExtraction, self).__init__("month", suffix_name="day")

    def process(self, data_source, field):
        data_source.data[self.get_feature_name(field)] = data_source.data[field].dt.month.astype('int')


class TimeFeatureMinuteExtraction(FeatureOperation):

    def __init__(self):
        super(TimeFeatureMinuteExtraction, self).__init__("minute", suffix_name="minute")

    def process(self, data_source, field):
        data_source.data[self.get_feature_name(field)] = data_source.data[field].dt.minute.astype('int')


class UserDefinedFunction(FeatureOperation):

    def __init__(self):
        super(UserDefinedFunction, self).__init__("fun", suffix_name="fun")
        self.fun = self.feature.fun

    def process(self, data_source, field):
        data_source.data[self.get_feature_name(field)] = data_source.data[field].apply(lambda x: self.fun(x), axis=1)


class DataTypeConversion(FeatureOperation):
    def __init__(self):
        super(DataTypeConversion, self).__init__("typeConv", suffix_name="ty")
        self.to_type = self.feature.dtype
        if self.to_type is None:
            raise ValueError("Cannot convert to type for feature " + self.feature.name)

    def process(self, data_source, field):
        # 转换数据类型
        data_source.data[self.get_feature_name(field)] = data_source.data[field].astype(self.to_type)
