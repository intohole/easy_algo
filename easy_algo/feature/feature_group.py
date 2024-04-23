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