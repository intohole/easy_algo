from easy_algo.util.manager import ModelFactory
from easy_algo.ml.lgb import *

ModelFactory.register("lgbBinaryAcc", LgbBinaryClassifierAcc)
ModelFactory.register("lgbBinaryAuc", LgbBinaryClassifierAuc)
ModelFactory.register("lgbRegressionMae", LgbRegressorMae)
ModelFactory.register("lgbRegressionMse", LgbRegressorMse)
