import processing
import pickle
from models.xgb import XGBModel
import yaml
import glob


def training(df, split_date, params, num_epoch, save_path=None, if_save=False):
    """
    训练+预测
    :param df: index=date/code, columns=lag
    :param split_date: str
    :param save_path:
    :param if_save:
    :return: y_pred: ndarray
    """
    # --------------- 数据准备 --------------------
    train_data, test_data = processing.split_train_test(df, split_date)
    X_train = train_data.iloc[:, :(train_data.shape[1]-1)]
    y_train = train_data.iloc[:, train_data.shape[1]-1]
    X_test = test_data.iloc[:, :(test_data.shape[1]-1)]
    y_test = test_data.iloc[:, test_data.shape[1]-1]

    # --------------- 模型定义 --------------------
    Model = XGBModel(params)

    # --------------- 训练 --------------------
    model, evals_result = Model.Train(X_train, y_train, num_epoch=num_epoch)
    print("Results:")
    for epoch, loss in enumerate(evals_result['train']['rmse'], 1):
        print(f"epoch {epoch}: RMSE = {loss}")

    # --------------- 预测 --------------------
    y_pred, loss_pred = Model.Predict(X_test, y_test)
    print(f'loss pred: {loss_pred}')

    # --------------- 保存模型 --------------------
    if if_save:
        pickle.dump(model, file=open(save_path, 'wb+'))

    return y_pred


def load_configs(config_folder):
    """
    加载指定文件夹中的所有 YAML 配置文件
    :param config_folder: 存放 YAML 配置文件的文件夹路径
    :return: 配置字典的列表
    """
    config_files = glob.glob(f"{config_folder}/*.yaml")
    configs = []
    for file in config_files:
        with open(file, 'r') as f:
            config = yaml.safe_load(f)
            configs.append(config)
    return configs


def model_to_factor(pred_data, df, multiindex, split_date, save_path=None, if_save=False):
    """
    将 pred_data 转换为因子值的格式， 方便进行回测
    :param pred_data: ndarray, 测试集的预测值
    :param df: dataframe, 整个数据集的x, y
    :param multiindex: list
    :param split_date: str
    :param save_path:
    :param if_save:
    :return:
    """
    # 过滤出大于 split_date 的 common_multiindex, df
    filtered_multiindex = [index for index in multiindex if index[0] > split_date]
    df_pred = df[df.index.get_level_values(0) > split_date]

    # 需要保存index与单因子的index是一致的 (将未上市/退市的股票去除)
    df_pred['pred'] = pred_data
    pred_data = df_pred[['pred']]
    new_factor = pred_data.reindex(filtered_multiindex)

    factor_dict = {date: new_factor.xs(date, level=0)['pred'] for date in new_factor.index.get_level_values(0).unique()}
    factor_dict = {date: factor_dict[date].sort_values(ascending=False) for date in factor_dict.keys()}

    if if_save:
        # 保存pkl
        processing.save_pkl(factor_dict, save_path)


def run_models(df, split_date, common_multiindex, num_epoch, configs, save_folder, save_model=False, save_factor=True):
    """
    使用多个配置文件进行多次模型训练
    :param df: 数据
    :param split_date: 数据拆分日期
    :param config_folder: 存放 YAML 配置文件的文件夹
    :param save_folder: 模型保存路径的文件夹
    """
    for i, config in enumerate(configs, 1):
        params = config['params']

        # 模型
        save_path = f'{save_folder}/{params["model_name"]}.pkl'
        print(f'\nTraining model {params["model_name"]} with configuration:')
        y_pred = training(df, split_date, params, num_epoch, save_path=save_path, if_save=save_model)
        print(f'Model {params["model_name"]} training completed.\n')

        # 因子
        save_path = f'{save_folder}/{params["factor_name"]}.pkl'
        model_to_factor(y_pred, df, common_multiindex, split_date, save_path=save_path, if_save=save_factor)
        print(f'factor {params["model_name"]} has been generated.\n')