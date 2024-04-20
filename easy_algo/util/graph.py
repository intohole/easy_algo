model_define = [
    [["group", "dense"], ["group", "dense"]],
    "concat",
    {"layer": "dense"},
    "output"
]
import tensorflow as tf


class LayerFactory:
    @staticmethod
    def create_model(layer_name, *args, **kwargs):
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


# 最小子粒度的list，只会返回一个output
#


def build_layer(layer, *args, **kwargs):
    if isinstance(layer, str):
        return LayerFactory.create_model(layer)
    elif isinstance(layer, dict):
        _layer = layer.copy()
        del _layer["layer"]
        return LayerFactory.create_model(layer["layer"], **_layer)
    else:
        raise NotImplementedError("Layer type not supported")


def build_model(model_define, input_data=None):
    output = input_data
    if isinstance(model_define, list):
        oa = []
        for index, item in enumerate(model_define):
            if isinstance(item, list):
                oa.append(build_model(item, output))
            elif isinstance(item, dict) or isinstance(item, str):
                if output is None:
                    output = build_layer(item)
                else:
                    output = build_layer(item)(output)
        return oa

    elif isinstance(model_define, dict) or isinstance(model_define, str):
        if output is None:
            output = build_layer(model_define)
        else:
            output = build_layer(model_define)(output)
    return output


model = build_model(model_define)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.summary()
