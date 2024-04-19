from enum import Enum


class TaskType(Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    BINARY = "binary"
    CLUSTERING = "clustering"
    RANKING = "ranking"



