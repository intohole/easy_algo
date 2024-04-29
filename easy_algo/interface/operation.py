class FeatureOperation:
    def __init__(self, name, suffix_name=None):
        self.name = name
        self._feature = None
        self._cover_name = False
        self.suffix_name = suffix_name
        self.col_name = None

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
        raise NotImplementedError("To be implemented")