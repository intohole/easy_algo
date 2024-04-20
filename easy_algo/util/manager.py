from easy_algo.ml.lgb import *


class ModelFactory:
    _registry = {
        "lgbBinaryAcc": LgbBinaryClassifierAcc,
        "lgbBinaryAuc": LgbBinaryClassifierAuc,
        "lgbRegressionMae": LgbRegressorMae,
        "lgbRegressionMse": LgbRegressorMse,
    }  # Registry to store custom operations

    @staticmethod
    def register(name, clazz):
        if name in ModelFactory._registry:
            raise TypeError("Duplicate operation named {}".format(name))
        """Register a custom feature operation class."""
        ModelFactory._registry[name] = clazz

    @staticmethod
    def create_model(name, *args, **kwargs):
        # Check if the operation is in the registry
        if name in ModelFactory._registry:
            return ModelFactory._registry[name](*args, **kwargs)

        raise ValueError("Unsupported feature operation name")


class LayerFactory(object):
    _registry = {

    }  # Registry to store custom operations

    @staticmethod
    def register(name, clazz):
        if name in LayerFactory._registry:
            raise TypeError("Duplicate operation named {}".format(name))
        """Register a custom feature operation class."""
        LayerFactory._registry[name] = clazz

    @staticmethod
    def create_model(name, *args, **kwargs):
        # Check if the operation is in the registry
        if name in LayerFactory._registry:
            return LayerFactory._registry[name](*args, **kwargs)

        raise ValueError("Unsupported feature operation name")
