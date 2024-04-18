from matplotlib.scale import LogTransform

from easy_algo.operation.operation import *


class FeatureOperationFactory:
    _registry = {
        "minMaxNor": MinMaxNormalization,
        "standardNor": Standardization,
        "equalWidthBucket": FeatureEqualWidth,
        "oneHot": OneHotEncoding,
        "labelEncode": LabelEncoding,
        "logTransform": LogTransform,
        "power": PowerTransformation,
        "modeMissing": ModeMissingValueImputation,
        "meanMissing": MeanMissingValueImputation,
        "medianMissing": MedianMissingValueImputation,
        "hour": TimeFeatureHourExtraction,
        "month": TimeFeatureMonthExtraction,
        "week": TimeFeatureWeekExtraction,
        "minute": TimeFeatureMinuteExtraction
    }  # Registry to store custom operations

    @staticmethod
    def register(name, clazz):
        if name in FeatureOperationFactory._registry:
            raise TypeError("Duplicate operation named {}".format(name))
        """Register a custom feature operation class."""
        FeatureOperationFactory._registry[name] = clazz

    @staticmethod
    def create_feature_operation(name, *args, **kwargs):
        # Check if the operation is in the registry
        if name in FeatureOperationFactory._registry:
            return FeatureOperationFactory._registry[name](*args, **kwargs)

        raise ValueError("Unsupported feature operation name")

    @staticmethod
    def create_operation(name, feature, *args, **kwargs):
        operation = FeatureOperationFactory.create_feature_operation(name, *args, **kwargs)
        operation._feature = feature
        return operation
