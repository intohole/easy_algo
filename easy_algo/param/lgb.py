from easy_algo.param import Param


class EnhancedLightGBMConfig(Param):
    """
    增强型LightGBM配置生成器
    """

    def __init__(self, task_type, boosting_type='gbdt', num_leaves=31, max_depth=-1,
                 learning_rate=0.1, n_estimators=100,
                 subsample_for_bin=200000, objective=None, class_weight=None,
                 min_split_gain=0., min_child_weight=1e-3, min_child_samples=20,
                 subsample=1., subsample_freq=0, colsample_bytree=1.,
                 reg_alpha=0., reg_lambda=0., random_state=None,
                 n_jobs=-1, silent=True, importance_type='split', **kwargs):
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
        self.task = task_type.value

        self.params = params = {
            'boosting_type': 'gbdt',
            'num_leaves': num_leaves,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'subsample_for_bin': subsample_for_bin,
            'objective': objective,  # 如果是二分类任务，可以设置为 'binary'
            'class_weight': class_weight,
            'min_split_gain': min_split_gain,
            'min_child_weight': min_child_weight,
            'min_child_samples': min_child_samples,
            'subsample': subsample,
            'subsample_freq': subsample_freq,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'random_state': random_state,
            'n_jobs': n_jobs,
            'silent': silent,
            'importance_type': importance_type,
        }

    def set_params_for_regression(self):
        """
        为回归任务设置参数
        """
        self.params['objective'] = 'regression'
        self.params['metric'] = 'rmse'

    def set_params_for_classification(self):
        """
        为分类任务设置参数
        """
        self.params['objective'] = 'binary'
        self.params['metric'] = 'binary_logloss'

    def set_params_for_ranking(self):
        """
        为排序任务设置参数
        """
        self.params['objective'] = 'lambdarank'
        self.params['metric'] = 'ndcg'

    def get_config(self):
        """
        获取最终的配置
        """
        if self.task == 'regression':
            self.set_params_for_regression()
        elif self.task == 'classification':
            self.set_params_for_classification()
        elif self.task == 'ranking':
            self.set_params_for_ranking()

        return self.params
