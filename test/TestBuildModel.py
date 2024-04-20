import tensorflow as tf


class LayerFactory:
    @staticmethod
    def create_layer(layer_name, *args, **kwargs):
        if layer_name == "dense":
            if len(args) == 0:
                args = [10]
            return tf.keras.layers.Dense(*args, **kwargs)
        elif layer_name == "group":
            if 'shape' not in kwargs:
                kwargs['shape'] = (3,)
            return tf.keras.layers.Input(*args, **kwargs)
        elif layer_name == "concat":
            return tf.keras.layers.Concatenate(*args, **kwargs)
        elif layer_name == "output":
            return tf.keras.layers.Dense(1, activation="sigmoid")
        else:
            raise ValueError(f"Unknown layer type: {layer_name}")


class ModelBuilder:

    def __init__(self, config):
        if config is None:
            raise ValueError("Config cannot be None")
        self.inputs = []
        self.outputs = []
        self.config = config
        self.model = None

    def parse_config(self, config):
        pass

    def get_layer(self, layer, *args, **kwargs):
        if isinstance(layer, str):
            return LayerFactory.create_layer(layer)
        elif isinstance(layer, dict):
            _layer = layer.copy()
            del _layer["layer"]
            return LayerFactory.create_layer(layer["layer"], **_layer)
        else:
            raise NotImplementedError("Layer type not supported")

    # model_define = [
    #
    #     [["group", "dense"], ["group", "dense"]],
    #     "concat",
    #     ["dense", "dense"],
    #     "concat",
    #     {"layer": "dense"},
    #     "output"
    # ]
    def _parse_list_config(self, config, input=None):
        for index, _config in enumerate(config):
            pass

    def build_model(self):
        for index, _layer_config in enumerate(self.config):
            if isinstance(_layer_config, dict) or isinstance(_layer_config, str):
                _layer = self.get_layer(_layer_config)
                if isinstance(_layer, tf.keras.layers.InputLayer):
                    self.inputs.append(_layer)
            if isinstance(_layer_config, list):
                _output = self._parse_list_config(_layer_config)


def build_model(model_define, input_data=None):
    model = tf.keras.Sequential()
    if isinstance(model_define, list):
        for item in model_define:
            if isinstance(item, list):
                model.add(build_model(item, input_data))
            elif isinstance(item, dict) or isinstance(item, str):
                layer = build_layer(item)
                model.add(layer)
                if input_data is not None:
                    model.add(layer(input_data))
                input_data = layer(input_data) if input_data is not None else None
    else:
        layer = build_layer(model_define)
        model.add(layer)
        if input_data is not None:
            model.add(layer(input_data))
        input_data = layer(input_data) if input_data is not None else None
    return model


# 使用示例
model_define = [

    [["group", "dense"], ["group", "dense"]],
    "concat",
    ["dense", "dense"],
    "concat",
    {"layer": "dense"},
    "output"
]

# 构建模型
model = build_model(model_define)

# 编译模型
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

import numpy as np

# 设置随机数种子以保证结果的可重复性
np.random.seed(0)

# 生成10维特征
num_features = 10
num_samples = 10000
feature_matrix1 = np.random.randn(num_samples, num_features)
feature_matrix2 = np.random.randn(num_samples, num_features)

# 生成标签，一半为1，一半为0
labels = np.random.randint(0, 2, num_samples)

model.fit(x=[feature_matrix1, feature_matrix2], y=labels, epochs=100)

# 打印模型结构
model.summary()
