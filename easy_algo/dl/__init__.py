from easy_algo.dl.layer.layer import *
from easy_algo.dl.layer.fm import *
from easy_algo.dl.layer.dnn import *
from easy_algo.util.manager import ModelFactory
from easy_algo.dl.layer.fm import FMLayer
import tensorflow as tf

ModelFactory.register("reluDense", tf.keras.layers.Dense, default_params={"units": 32, "activation": "relu"})
ModelFactory.register("dense", tf.keras.layers.Dense, default_params={"units": 32, "activation": "relu"})
ModelFactory.register("flatten", tf.keras.layers.Flatten)
ModelFactory.register("concat", tf.keras.layers.Concatenate)
ModelFactory.register("dropout", tf.keras.layers.Dropout, default_params={"rate": 0.3})
ModelFactory.register("add", tf.keras.layers.Add)
ModelFactory.register("average", tf.keras.layers.Average)
ModelFactory.register("subtract", tf.keras.layers.Subtract)
ModelFactory.register("singleSigmoid", tf.keras.layers.Dense, default_params={"units": 1, "activation": "sigmoid"})
ModelFactory.register("sigmoidDense", tf.keras.layers.Dense, default_params={"units": 32, "activation": "sigmoid"})
ModelFactory.register("tanhDense", tf.keras.layers.Dense, default_params={"units": 32, "activation": "tanh"})
ModelFactory.register("softmaxDense", tf.keras.layers.Dense, default_params={"units": 32, "activation": "softmax"})
ModelFactory.register("fmLayer", FMLayer, default_params={})
ModelFactory.register("dnn", DeepLayer, default_params={"hidden_units": [128, 64, 32, 16], "activation": "relu"})
