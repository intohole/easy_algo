import pandas as pd
from easy_algo.util.constant import FeatureType
from collections import defaultdict


class DataSource(object):

    def __init__(self, data):
        self.data = None

    def select(self, condition):
        raise NotImplementedError

    def filter(self, condition):
        raise NotImplementedError

    def update(self, condition, value):
        raise NotImplementedError

    def drop(self, condition):
        raise NotImplementedError

    def get_xy(self, features):
        raise NotImplementedError

    def dtypes(self):
        pass

