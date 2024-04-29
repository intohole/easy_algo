class FeatureGroup:
    def __init__(self, group_name, features=None, label=False):
        self.name = group_name
        self.shape = None
        self._features = features if features is not None else []
        self.update_shape()
        self.label = label

    @property
    def features(self):
        """
        获取features属性的值。
        """
        return self._features

    def append_feature(self, feature):
        self._features.append(feature)
        self.update_shape()

    def del_feature(self, feature):
        self._features.remove(feature)
        self.update_shape()

    def update_shape(self):
        self.shape = sum(_.shape for _ in self.features)

    @features.setter
    def features(self, value):
        """
        更新features属性时，自动更新shape属性。
        """
        self._features = value
        self.update_shape()
