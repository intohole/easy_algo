from sklearn.metrics import mean_squared_error, roc_auc_score, mean_absolute_error

from easy_algo.interface.model import BaseModel
from easy_algo.param.lgb import EnhancedLightGBMConfig
from easy_algo.util.constant import TaskType
import lightgbm as lgb


class LgbBinaryClassifierBuilder(BaseModel):

    def build(self):
        self.model = lgb.LGBMClassifier(
            EnhancedLightGBMConfig(TaskType.CLASSIFICATION).get_config()
        )


class LgbRegressorBuilder(BaseModel):
    def build(self):
        self.model = lgb.LGBMRegressor(
            EnhancedLightGBMConfig(TaskType.REGRESSION).get_config()
        )


class LgbTrainer(BaseModel):

    def train(self, x, y):
        self.model = lgb.train(x, y)


class LgbPredict(BaseModel):

    def predict(self, x):
        self.model.predict(x)


class LgbBinaryClassifierAccEvaluator(BaseModel):
    def evaluate(self, x, y):
        # 这里使用准确率作为评估指标，可以根据需要更改
        predictions = self.model.predict(x)
        accuracy = (predictions == y).mean()
        print(f"Accuracy: {accuracy}")
        return accuracy


class LgbRegressorMseEvaluator(BaseModel):
    def evaluate(self, x, y):
        # 这里使用均方误差作为评估指标，可以根据需要更改
        mse = mean_squared_error(y, self.model.predict(x))
        print(f"Mean Squared Error: {mse}")
        return mse


class LgbBinaryClassifierAucEvaluator(BaseModel):
    def evaluate(self, x, y):
        # 计算AUC
        predictions_proba = self.model.predict_proba(x)[:, 1]  # 获取正类的预测概率
        auc = roc_auc_score(y, predictions_proba)
        print(f"AUC: {auc}")
        return auc


class LgbRegressorMaeEvaluator(BaseModel):
    def evaluate(self, x, y):
        # 计算MAE
        mae = mean_absolute_error(y, self.model.predict(x))
        print(f"Mean Absolute Error: {mae}")
        return mae


class LgbBinaryClassifierAcc(LgbBinaryClassifierAccEvaluator, LgbTrainer, LgbPredict, LgbBinaryClassifierAccEvaluator):
    pass


class LgbBinaryClassifierAuc(LgbBinaryClassifierAccEvaluator, LgbTrainer, LgbPredict, LgbBinaryClassifierAucEvaluator):
    pass


class LgbRegressorMse(LgbRegressorMseEvaluator, LgbTrainer, LgbPredict, LgbRegressorMseEvaluator):
    pass


class LgbRegressorMae(LgbRegressorMseEvaluator, LgbTrainer, LgbPredict, LgbRegressorMaeEvaluator):
    pass
