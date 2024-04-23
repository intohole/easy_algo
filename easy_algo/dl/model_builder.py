import tensorflow as tf
from tensorflow.keras.models import Model
from easy_algo.util.manager import ModelFactory


class ModelBuilder:

    def __init__(self, config, schema=None):
        if config is None:
            raise ValueError("Config cannot be None")
        self.input_layer_map = {_: tf.keras.layers.Input(shape=(10,)) for _ in ["group"]}
        self.config = config
        self.schema = schema
        if schema is not None:
            self.init_schema()
        self.outputs = self.build_model()
        self.model = Model(inputs=[self.input_layer_map[_] for _ in ["group"]], outputs=self.outputs)

    def init_schema(self):
        pass

    def get_layer(self, layer, *args, **kwargs):
        if isinstance(layer, str) and layer in self.input_layer_map:
            return self.input_layer_map[layer]
        if isinstance(layer, str):
            return ModelFactory.create(layer)
        elif isinstance(layer, dict):
            _layer = layer.copy()
            del _layer["layer"]
            return ModelFactory.create(layer["layer"], **_layer)
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
