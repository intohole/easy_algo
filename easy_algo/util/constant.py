from enum import Enum


class TaskType(Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    BINARY = "binary"
    CLUSTERING = "clustering"
    RANKING = "ranking"


class FeatureType(Enum):
    Feature = "feature"
    Label = "label"
    Other = "other"
