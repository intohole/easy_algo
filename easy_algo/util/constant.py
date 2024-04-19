from enum import Enum


class TaskType(Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"
    RANKING = "ranking"
