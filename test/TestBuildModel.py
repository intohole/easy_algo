from easy_algo.dl.model import *

from easy_algo.feature.schema import FeatureSchema
from easy_algo.dl.dl import DL

if __name__ == "__main__":
    schema = FeatureSchema.build_feature_schema(set(), ['feature1'], [int])

    model_define = [

        [["group0", "dense"], ["group0", "dense"]],
        "concat",
        "dnn",
        {"layer": "dense"},
        "singleSigmoid"
    ]

    # 构建模型
    dl = DL(model_define, schema, 'binaryAcc')

    # 编译模型

    import numpy as np

    # 设置随机数种子以保证结果的可重复性
    np.random.seed(0)

    # 生成10维特征
    num_features = 1
    num_samples = 10000
    feature_matrix1 = np.random.randn(num_samples, num_features)
    feature_matrix2 = np.random.randn(num_samples, num_features)

    # 生成标签，一半为1，一半为0
    labels = np.random.randint(0, 2, num_samples)

    dl.model.summary()

    dl.fit(x=[feature_matrix1], y=labels)

    # 打印模型结构
