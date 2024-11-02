import lightgbm as lgb
import pickle
from sklearn.metrics import mean_squared_error


class LGBModel:
    def __init__(self, params):
        self.max_depth = params['max_depth']   # 树的最大深度，控制模型复杂度
        self.learning_rate = params['eta']   # 学习率，减少每步的权重，防止过拟合
        self.num_leaves = params['num_leaves']   # 树的最大叶节点数, 理想情况下应小于 2^(max_depth)
        self.bagging_fraction = params['bagging_fraction']   # 用于行采样（随机抽取数据的百分比）来加快训练速度并减少过拟合
        self.bagging_freq = params['bagging_freq']   # 与 bagging_fraction 配合使用，指定执行行采样的频率。例如，bagging_freq=5 表示每 5 轮执行一次采样
        self.feature_fraction = params['feature_fraction']   # 用于特征采样，即每棵树随机选择部分特征来分裂。常用值在 0.6-0.9
        self.lambda_l1 = params['lambda_l1']
        self.seed = params['seed']
        self.model = None

    def Train(self, X, y, num_epoch=100):
        dtrain = lgb.Dataset(X, label=y)

        params = {
            'objective': 'regression',  # 回归任务
            'metric': 'mse',   # 回归任务的目标函数
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'num_leaves': self.num_leaves,
            'bagging_fraction': self.bagging_fraction,
            'bagging_freq': self.bagging_freq,
            'feature_fraction': self.feature_fraction,
            'lambda_l1': self.lambda_l1,
            'seed': self.seed
        }

        evals_result = {}  # 用来存储评估结果
        lgb_model = lgb.train(params, dtrain, num_epoch)
        self.model = lgb_model
        return lgb_model, evals_result

    def Predict(self, X, y, model_path=None):
        dtest = lgb.Dataset(X, label=y)

        if self.model is None:
            # 读取模型
            with open(model_path,'rb') as file:
                self.model = pickle.load(file)
        y_pred = self.model.predict(dtest)
        mse = mean_squared_error(y, y_pred)
        return y_pred, mse