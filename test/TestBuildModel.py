from easy_algo.dl.model import *

from easy_algo.feature.schema import FeatureSchema
from easy_algo.dl.dl import DL
from easy_algo.process.processor import PandasProcessor

if __name__ == "__main__":
    import pandas as pd

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
    # 创建一个包含三列数据的DataFrame，最后一列为0和1
    data = {'A': np.random.randint(1, 1000, 1000), 'B': np.random.randint(1, 1000, 1000),
            'C': np.random.choice([0, 1], 1000)}

    df = pd.DataFrame(data)

    a = df['A']
    print(a[:10])
    model_define = [

        [["group0", "dense"], ["group0", "dense"]],
        "concat",
        "dnn",
        {"layer": "dense"},
        "singleSigmoid"
    ]
    processor = PandasProcessor(model_config=model_define, data_frame=df, features=['A'], labels=['C'],
                                trainer='binaryAcc')

    processor.process()
    # schema = FeatureSchema.build_feature_schema(set(), ['feature1'], [int])
    #
    # # 构建模型
    # dl = DL(model_define, schema, 'binaryAcc')
    #
    # # 编译模型
    #
    # dl.model.summary()
    #
    # dl.fit(x=[feature_matrix1], y=labels)

    # 打印模型结构
