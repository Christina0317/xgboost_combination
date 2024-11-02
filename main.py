import processing
from config import *
import pickle
import os
import method
import pandas as pd


if __name__ == '__main__':
    startdate = int(20190101)
    enddate = int(20241231)

    gd = processing.GetData(factor_names, daily_stock_path, startdate, enddate)

    return_data = gd.get_return_data()

    lag_day = 5  # 考虑过去lag_day的特征

    feature_data, common_multiindex = gd.get_factor_data(lag_day)

    df = gd.combine_data(feature_data, return_data)

    config_folder = '/Users/hjx/Documents/projects/xgboost_combination/configs'
    configs = method.load_configs(config_folder)
    for config in configs:
        config['params']['model_name'] = f'model_lgb_lag_{lag_day}_depth_{config["params"]["max_depth"]}_eta_{config["params"]["eta"]}'
        config['params']['factor_name'] = f'factor_lgb_lag_{lag_day}_depth_{config["params"]["max_depth"]}_eta_{config["params"]["eta"]}'

    save_folder = '/Users/hjx/Documents/projects/xgboost_combination/res'
    method.run_models(df, split_date, common_multiindex, num_epoch, configs, save_folder, save_model=False, save_factor=True)
