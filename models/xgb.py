import xgboost as xgb
import pickle
from sklearn.metrics import mean_squared_error


class XGBModel:
    def __init__(self, params):
        self.max_depth = params['max_depth']   # 树的最大深度，控制模型复杂度
        self.eta = params['eta']   # 学习率，减少每步的权重，防止过拟合
        self.subsample = params['subsample']   # 训练每棵树时使用的数据比例，防止过拟合
        self.colsample_bytree = params['colsample_bytree']   # 建立每棵树时列的采样比率
        self.n_jobs = params['n_jobs']
        self.verbosity = params['verbosity']
        self.seed = params['seed']
        self.model = None

    def Train(self, X, y, num_epoch=100):
        # n_estimators是树的数量，max_depth是树的最大深度
        dtrain = xgb.DMatrix(X, label=y)

        params = {
            'objective': 'reg:squarederror',  # 回归任务的目标函数
            'max_depth': self.max_depth,
            'eta': self.eta,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'n_jobs': self.n_jobs,
            'verbosity': self.verbosity,
            'seed': self.seed
        }

        evals_result = {}  # 用来存储评估结果
        xgb_model = xgb.train(params, dtrain, num_epoch, evals=[(dtrain, 'train')], evals_result=evals_result,
                              verbose_eval=True)
        self.model = xgb_model
        return xgb_model, evals_result

    def Predict(self, X, y, model_path=None):
        dtest = xgb.DMatrix(X, label=y)

        if self.model is None:
            # 读取模型
            with open(model_path,'rb') as file:
                self.model = pickle.load(file)
        y_pred = self.model.predict(dtest)
        mse = mean_squared_error(y, y_pred)
        return y_pred, mse