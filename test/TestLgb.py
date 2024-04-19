from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from easy_algo.ml.lgb import LgbBinaryClassifierAcc

model = LgbBinaryClassifierAcc()
# 1. 数据准备
# 创建模拟的二分类数据集
X, y = make_classification(n_samples=10000, n_features=10, n_classes=2, random_state=42)
print(X[:100])
print(y[:100])
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.build()
model.train(X_train, y_train)
model.evaluate(X_test, y_test)