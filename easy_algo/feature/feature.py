from enum import Enum

FEATURE_TYPE = Enum("FeatureType",("feature","label","other"))

class Feature(object):
    def __init__(self, feature_name, value_type= FEATURE_TYPE.feature, feature_process=[],cover_name = False):
        self.value_type = value_type
        self.feature_name = feature_name
        self.type = value_type
        self.feature_process = feature_process
        self.cover_name = cover_name
        self.processors = []
        self.train_able = True


class DenseFeature(Feature):
    def __init__(self, feature_name, value_type='feature', feature_process=[]):
        super(DenseFeature, self).__init__(feature_name, value_type, feature_process)
        self.bucket_num = None
        self.condition_map = {}




class CategoryFeature(Feature):
    def __init__(self, feature_name, voca_size, out_dim, value_type='feature', feature_process=[]):
        super(CategoryFeature, self).__init__(feature_name, value_type, feature_process)
        self.voca_size = voca_size
        self.out_dim = out_dim
