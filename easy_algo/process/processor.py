from sklearn.model_selection import train_test_split
from random import randint

from easy_algo.interface.processor import Processor
from easy_algo.util.manager import ModelFactory
from easy_algo.dl.dl import DL
from easy_algo.source.panda_source import PandaDataSource
from easy_algo.util.constant import Event
from easy_algo.feature.schema import FeatureSchema


class PandasProcessor(Processor):

    def __init__(self, model_config=None,
                 model_name=None,
                 data_frame=None,
                 data_path=None,
                 schema=None,
                 feature_columns=None,
                 test_size=0.2,
                 random_state=randint(0, 100),
                 trainer=None,
                 learn_rate=None,
                 categories=None,
                 labels=None,
                 *args,
                 **kwargs):
        super().__init__(feature_columns)
        if categories is None:
            categories = []
        self.data_path = data_path
        self.data = PandaDataSource(data_frame=data_frame, data_path=data_path)
        self.test_size = test_size
        self.random_state = random_state
        self.model_config = model_config
        self._schema = schema
        self.model_name = model_name
        self._x_train = None
        self._y_train = None
        self._x_test = None
        self._y_test = None
        self._categories = categories
        self._label = labels
        self._learn_rate = learn_rate
        self._trainer = trainer
        self.feature_columns = self.schema.features if feature_columns is None else feature_columns
        self.model = self.build_model(*args, **kwargs)

    @property
    def schema(self):
        if self._schema is None:
            self._schema = FeatureSchema.build_feature_schema(self._categories, self._label, self.data.columns,
                                                              self.data.dtypes)
        return self._schema

    def build_model(self, *args, **kwargs):
        if self.model_name is not None:
            return ModelFactory.create(self.model_name, *args, **kwargs)

        elif self.model_config is not None:
            if isinstance(self.model_config, list):
                return DL(self.model_config, self.schema, trainer=self._trainer, *args, **kwargs)
        raise TypeError("Model type must be str or dict")

    def get_columns(self):
        return self.data.columns

    def split_data(self):
        X, y = self.get_xy()
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size,
                                                            random_state=self.random_state)
        return x_train, y_train, x_test, y_test

    def train(self):
        self.model.build()
        self.model.fit(self.x_train, self.y_train)

    @property
    def x_train(self):
        if self._x_train is None:
            self._x_train, self._y_train, self._x_test, self._y_test = self.split_data()
        else:
            return self._x_train

    @property
    def y_train(self):
        if self._y_train is None:
            self._x_train, self._y_train, self._x_test, self._y_test = self.split_data()
        return self._y_train

    @property
    def x_test(self):
        if self._x_test is None:
            self._x_train, self._y_train, self._x_test, self._y_test = self.split_data()
        return self._x_test

    @property
    def y_test(self):
        if self._y_test is None:
            self._x_train, self._y_train, self._x_test, self._y_test = self.split_data()
        return self._y_test

    def evaluate(self):
        self.model.evaluate(self.x_test, self.y_test)

    def get_xy(self, test_condition=None):
        # 获取每个group的字段名称
        feature_fields, label_fields = self._schema.generate_feature_names()
        x = []
        y = []
        for _fields in feature_fields:
            x.append(self.data.data_frame[_fields].values)
        for _fields in label_fields:
            y.append(self.data.data_frame[_fields].values)
        return x, y

    def __getitem__(self, item):
        pass

    def notify(self, event, *args, **kwargs):
        if event == Event.PROCESS:
            feature, processor = args[0], args[1]
            processor.process(self.data, feature.col_name)

        if event == Event.GROUP_CHANGE:
            feature, old, new = args
            # todo 实现
            self._schema.update_feature_group(old, new, feature)

    def process(self):
        # 特征处理
        # 假设feature_processor是用于处理特征的函数
        for feature in self.feature_columns:
            for processor in feature.processors:
                processor.process(self.data, feature.col_name)

        # 数据切分
        X, y = self.get_xy()
        x_train, x_test, y_train, y_test = train_test_split(X[0], y[0], test_size=self.test_size,
                                                            random_state=self.random_state)
        self.model.fit(x_train, y_train)
        self.model.evaluate(x_test, y_test)
