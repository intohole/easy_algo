from easy_algo.operation.operation import *



class FeatureOperationFactory:
    _registry = {}  # Registry to store custom operations

    @staticmethod
    def register(name, clazz):
        """Register a custom feature operation class."""
        FeatureOperationFactory._registry[name] = clazz

    @staticmethod
    def create_feature_operation(name, *args, **kwargs):
        # Check if the operation is in the registry
        if name in FeatureOperationFactory._registry:
            return FeatureOperationFactory._registry[name](*args, **kwargs)

        if name == "minMaxNor":
            return MinMaxNormalization()
        elif name == "standardNor":
            return Standardization(*args, **kwargs)
        elif name == "equalWidthBucket":
            return FeatureEqualWidth(*args, **kwargs)
        elif name == "oneHot":
            return OneHotEncoding()
        elif name == "labelEncode":
            return LabelEncoding()
        elif name == "logTransform":
            return LogTransformation()
        elif name == "power":
            return PowerTransformation()
        elif name == "modeImputation":
            return ModeMissingValueImputation()
        elif name == "meanImputation":
            return MeanMissingValueImputation()
        elif name == "medianImputation":
            return MedianMissingValueImputation()
        elif name == "hour":
            return TimeFeatureHourExtraction()
        elif name == "month":
            return TimeFeatureMonthExtraction()
        elif name == "minute":
            return TimeFeatureMinuteExtraction()
        else:
            raise ValueError("Unsupported feature operation name")

    @staticmethod
    def create_operation(name, feature, *args, **kwargs):
        operation = FeatureOperationFactory.create_feature_operation(name, *args, **kwargs)
        operation._feature = feature
        return operation