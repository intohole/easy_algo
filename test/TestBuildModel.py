import tensorflow as tf
from keras import Model


class LayerFactory:
    @staticmethod
    def create_layer(layer_name, *args, **kwargs):
        if layer_name == "dense":
            if len(args) == 0:
                args = [10]
            kwargs.update({"activation": 'relu'})
            return tf.keras.layers.Dense(*args, **kwargs)
        elif layer_name == "group":
            if 'shape' not in kwargs:
                kwargs['shape'] = (3,)
            return tf.keras.layers.Input(*args, **kwargs)
        elif layer_name == "concat":
            kwargs['axis'] = -1
            return tf.keras.layers.Concatenate(*args, **kwargs)
        elif layer_name == "output":
            return tf.keras.layers.Dense(1, activation="sigmoid")
        else:
            raise ValueError(f"Unknown layer type: {layer_name}")


class ModelBuilder:

    def __init__(self, config):
        if config is None:
            raise ValueError("Config cannot be None")
        self.input_layer_map = {_:tf.keras.layers.Input(shape=(10,)) for _ in ["group"]}
        self.config = config
        self.outputs = self.build_model()
        self.model = Model(inputs = [self.input_layer_map[_] for _ in ["group"]] , outputs=self.outputs)

    def get_layer(self, layer, *args, **kwargs):
        if isinstance(layer,str) and layer in self.input_layer_map:
            return self.input_layer_map[layer]
        if isinstance(layer, str):
            return LayerFactory.create_layer(layer)
        elif isinstance(layer, dict):
            _layer = layer.copy()
            del _layer["layer"]
            return LayerFactory.create_layer(layer["layer"], **_layer)
        else:
            raise NotImplementedError("Layer type not supported")

    def _all_is_iter(self, config):
        if isinstance(config, list):
            for _ in config:
                if not isinstance(_, list):
                    return False
            return True
        else:
            return False

    def _parse_list_config(self, config, input_layer=None):
        outputs = []
        if not self._all_is_iter(config):
            raise TypeError("use list config ,please config [[\"dese\"],[\"dense\"]] like this,list only contain "
                            "list")
        for index, _config in enumerate(config):
            _layer = input_layer
            for _in in _config:
                if isinstance(_in, (str, dict)):
                    _layer = self._layer_create(self.get_layer(_in), _layer)
                else:
                    raise NotImplementedError("Layer type not supported")
            outputs.append(_layer)
        return outputs

    def _layer_create(self, current_layer, input_layer=None):
        if input_layer is None:
            return current_layer
        return current_layer(input_layer)

    def build_model(self):
        _layer = None
        outputs = None
        for index, _layer_config in enumerate(self.config):
            if isinstance(_layer_config, (dict, str)):
                _layer = self._layer_create(self.get_layer(_layer_config), _layer)
            elif isinstance(_layer_config, list):
                _layer = self._parse_list_config(config=_layer_config, input_layer=_layer)
            else:
                raise TypeError("Layer type not supported")
            if index == len(self.config) - 1:
                outputs = _layer
        return outputs


if __name__ == "__main__":

    input = tf.keras.layers.Input(shape=(3,))
    x = tf.keras.layers.Dense(10, activation="relu")(input)
    y = tf.keras.layers.Dense(10, activation="relu")(input)
    x = tf.keras.layers.Concatenate(axis=-1)([x, y])
    x = tf.keras.layers.Dense(10, activation="relu")(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = Model(inputs=input, outputs=x)
    model.summary()

    model_define = [

        [["group", "dense"], ["group", "dense"]],
        "concat",
        "dense",
        {"layer": "dense"},
        "output"
    ]

    # 构建模型
    model = ModelBuilder(model_define)

    model = model.model
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

model.fit(x=[feature_matrix1], y=labels, epochs=100)

# 打印模型结构
model.summary()
