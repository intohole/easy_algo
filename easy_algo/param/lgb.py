from easy_algo.param import Param

class EnhancedLightGBMConfig(Param):
    """
    增强型LightGBM配置生成器
    """

    def __init__(self, task_type, boosting_type='gbdt', num_leaves=31, learning_rate=0.1, n_estimators=100,
                 feature_fraction=1.0, bagging_fraction=1.0, bagging_freq=0, bagging_seed=3, feature_fraction_seed=2,
                 min_data_in_leaf=20, max_depth=-1, lambda_l1=0.0, lambda_l2=0.0, is_unbalance=False):
        """
        初始化基础配置
        :param task_type: 任务类型，如'regression', 'classification', 'ranking'
        :param boosting_type: 提升类型
        :param num_leaves: 叶子数
        :param learning_rate: 学习率
        :param n_estimators: 树的数量
        :param feature_fraction: 特征采样比例
        :param bagging_fraction: Bagging采样比例
        :param bagging_freq: Bagging频率
        :param bagging_seed: Bagging随机种子
        :param feature_fraction_seed: 特征采样随机种子
        :param min_data_in_leaf: 叶子上的最小数据量
        :param max_depth: 树的最大深度
        :param lambda_l1: L1正则化系数
        :param lambda_l2: L2正则化系数
        :param is_unbalance: 是否为不平衡数据
        """
        super().__init__(task_type)
        self.config = {
            'task': task_type,
            'boosting_type': boosting_type,
            'num_leaves': num_leaves,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'feature_fraction': feature_fraction,
            'bagging_fraction': bagging_fraction,
            'bagging_freq': bagging_freq,
            'bagging_seed': bagging_seed,
            'feature_fraction_seed': feature_fraction_seed,
            'min_data_in_leaf': min_data_in_leaf,
            'max_depth': max_depth,
            'lambda_l1': lambda_l1,
            'lambda_l2': lambda_l2,
            'is_unbalance': is_unbalance
        }

    def set_params_for_regression(self):
        """
        为回归任务设置参数
        """
        self.config['objective'] = 'regression'
        self.config['metric'] = 'rmse'

    def set_params_for_classification(self):
        """
        为分类任务设置参数
        """
        self.config['objective'] = 'binary'
        self.config['metric'] = 'binary_logloss'

    def set_params_for_ranking(self):
        """
        为排序任务设置参数
        """
        self.config['objective'] = 'lambdarank'
        self.config['metric'] = 'ndcg'

    def get_config(self):
        """
        获取最终的配置
        """
        if self.config['task'] == 'regression':
            self.set_params_for_regression()
        elif self.config['task'] == 'classification':
            self.set_params_for_classification()
        elif self.config['task'] == 'ranking':
            self.set_params_for_ranking()
        return self.config