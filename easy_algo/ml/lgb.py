import sys

from sklearn.metrics import mean_squared_error, roc_auc_score, mean_absolute_error

from easy_algo.interface.model import BaseModel
from easy_algo.param.lgb import EnhancedLightGBMConfig
from easy_algo.util.constant import TaskType
import lightgbm as lgb
import random

from sklearn.linear_model import LogisticRegression


class LrBinaryClassifierBuilder(BaseModel):

    def __init__(self, max_iter=None, random_state=None):
        super(LrBinaryClassifierBuilder, self).__init__()
        self.max_iter = max_iter if max_iter is not None else 100
        self.random_state = random_state if random_state is not None else random.randint(0, 100)

    def build(self):
        # 默认参数可以根据需要进行调整
        self._model = LogisticRegression(max_iter=1000, random_state=42)


class LrTrainer(BaseModel):

    def train(self, x, y):
        self._model.fit(x, y)


class LrPredict(BaseModel):

    def predict(self, x):
        return self._model.predict(x)


class LgbBinaryClassifierBuilder(BaseModel):

    def build(self):
        self._model = lgb.LGBMClassifier(
            **EnhancedLightGBMConfig(TaskType.CLASSIFICATION).get_config()
        )


class LgbMultiClassifierBuilder(BaseModel):

    def build(self):
        self._model = lgb.LGBMClassifier(**EnhancedLightGBMConfig(TaskType.MULTICLASS_CLASSIFICATION).get_config())


class LgbRegressorBuilder(BaseModel):
    def build(self):
        self._model = lgb.LGBMRegressor(
            **EnhancedLightGBMConfig(TaskType.REGRESSION).get_config()
        )


class FitTrainer(BaseModel):

    def train(self, x, y):
        self._model.fit(x, y)


class ModelPredict(BaseModel):

    def predict(self, x):
        self._model.predict(x)


class LgbBinaryClassifierAccEvaluator(BaseModel):
    def evaluate(self, x, y):
        # 这里使用准确率作为评估指标，可以根据需要更改
        predictions = self.predict(x)
        accuracy = (predictions == y).mean()
        print(f"Accuracy: {accuracy}")
        return accuracy


class LgbRegressorMseEvaluator(BaseModel):
    def evaluate(self, x, y):
        # 这里使用均方误差作为评估指标，可以根据需要更改
        mse = mean_squared_error(y, self.predict(x))
        print(f"Mean Squared Error: {mse}")
        return mse


class LgbBinaryClassifierAucEvaluator(BaseModel):
    def evaluate(self, x, y):
        # 计算AUC
        predictions_proba = self._model.predict_proba(x)[:, 1]  # 获取正类的预测概率
        auc = roc_auc_score(y, predictions_proba)
        print(f"AUC: {auc}")
        return auc


class LgbRegressorMaeEvaluator(BaseModel):
    def evaluate(self, x, y):
        # 计算MAE
        mae = mean_absolute_error(y, self.predict(x))
        print(f"Mean Absolute Error: {mae}")
        return mae


class LgbBinaryClassifierAcc(LgbBinaryClassifierBuilder, FitTrainer, ModelPredict, LgbBinaryClassifierAccEvaluator):
    pass


class LgbBinaryClassifierAuc(LgbBinaryClassifierBuilder, FitTrainer, ModelPredict, LgbBinaryClassifierAucEvaluator):
    pass


class LgbRegressorMse(LgbRegressorBuilder, FitTrainer, ModelPredict, LgbRegressorMseEvaluator):
    pass


class LgbRegressorMae(LgbRegressorBuilder, FitTrainer, ModelPredict, LgbRegressorMaeEvaluator):
    pass
