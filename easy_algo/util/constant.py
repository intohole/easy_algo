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


class Optimizer(Enum):
    ADAM = 'adam'
    SGD = 'sgd'
    RMSPROP = 'rmsprop'
    ADAGRAD = 'adagrad'


class Loss(Enum):
    MEAN_SQUARED_ERROR = 'mean_squared_error'
    BINARY_CROSSENTROY = 'binary_crossentropy'
    SOFTMAX_CROSS_ENTROPY = 'softmax_cross_entropy'
    CATEGORICAL_CROSS_ENTROPY = 'categorical_cross_entropy'
    HINGE = 'hinge'
    MAE = 'mean_absolute_error'


class Metrics(Enum):
    ACC = 'accuracy'
    SQUARED_ERROR = 'mean_squared_error'
    CATEGORICAL_ACCURACY = 'categorical_accuracy'
    TOP_K_CATEGORICAL_ACCURACY = 'top_k_categorical_accuracy'
    AUC = 'auc'
    SENSITIVITY = 'sensitivity'
    SPECIFICITY = 'specificity'
    POSITIVE_PREDICTIVE_VALUE = 'positive_predictive_value'
    NEGATIVE_PREDICTIVE_VALUE = 'negative_predictive_value'
    PRECISION = 'precision'
    RECALL = 'recall'
    F1_SCORE = 'f1_score'
    MAE = 'mae'
    MSE = 'mse'


class Callback(Enum):
    EARLY_STOPPING = 'early_stopping'
    MODEL_CHECKPOINT = 'model_checkpoint'
    REDUCE_LR_ON_PLATEAU = 'reduce_lr_on_plateau'


class ModelType(Enum):
    DL = "dl"
    ML = "ml"
