from easy_algo.ml.lgb import *
import easy_algo.dl
import easy_algo.ml


class ModelFactory(object):
    _registry = {

    }  # Registry to store custom operations

    @staticmethod
    def register(name, clazz):
        if name in ModelFactory._registry:
            raise TypeError("Duplicate operation named {}".format(name))
        """Register a custom feature operation class."""
        ModelFactory._registry[name] = clazz

    @staticmethod
    def create(name, *args, **kwargs):
        # Check if the operation is in the registry
        if name in ModelFactory._registry:
            return ModelFactory._registry[name](*args, **kwargs)

        raise ValueError("Unsupported feature operation name")
