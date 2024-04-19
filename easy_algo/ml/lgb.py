import lightgbm as lgb
from easy_algo.interface.model import Model



class LgbBinaryClassifier():

    def __init__(self, model: lgb.LGBMClassifier):
        super().__init__()
        self.boosting_type = 'gbdt'
        self.objective = 'binary'
        self.metric = 'auc'
        self.eval_metric = 'auc'
        self.eval_fold = 5
        self.learning_rate = 0.01
        self.min_child_weight = 1
        self.num_leaves = 31
        self.feature_fraction = 0.8
        self.bagging_fraction = 0.8
        self.bagging_freq = 5
        self.verbose = 0



class LGB(Model):

    def __init__(self,):
        self.model = lgb.LGBMClassifier(
            boosting_type='gbdt',
            objective='binary',
        )

