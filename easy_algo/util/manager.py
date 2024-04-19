from easy_algo.ml.lgb import *


class Factory:
    _registry = {
        "lgbBinaryAcc": LgbBinaryClassifierAcc,
        "lgbBinaryAuc": LgbBinaryClassifierAuc,
        "lgbRegressionMae": LgbRegressorMae,
        "lgbRegressionMse": LgbRegressorMse,
    }  # Registry to store custom operations

    @staticmethod
    def register(name, clazz):
        if name in Factory._registry:
            raise TypeError("Duplicate operation named {}".format(name))
        """Register a custom feature operation class."""
        Factory._registry[name] = clazz

    @staticmethod
    def create_model(name, *args, **kwargs):
        # Check if the operation is in the registry
        if name in Factory._registry:
            return Factory._registry[name](*args, **kwargs)

        raise ValueError("Unsupported feature operation name")
