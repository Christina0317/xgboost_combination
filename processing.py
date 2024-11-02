import pandas as pd
import os
import pickle
from functools import reduce
from config import *


def save_pkl(factor_dict, name):
    print('The file ' + str(name) + ' has been saved')
    if os.path.exists(name):
        os.remove(name)
    with open(name, 'wb') as file:
        pickle.dump(factor_dict, file)


def filling_nan(factor_dict, method):
    if method == 'mean':
        factor_dict_filled = {}
        for date, series in factor_dict.items():
            mean = series.mean()
            series_filled = series.fillna(mean)
            factor_dict_filled[date] = series_filled
        return factor_dict_filled


def get_multiindex(factor_dict):
    df_list = []
    for date, series in factor_dict.items():
        df_temp = series.to_frame(name='factor_value')
        df_temp['date'] = date
        df_list.append(df_temp)

    df_combined = pd.concat(df_list)
    df_combined.set_index(['date', df_combined.index], inplace=True)
    df_combined.index.names = ['date', 'code']
    return df_combined.index


def split_train_test(df, split_date):
    train_df = df[df.index.get_level_values(0) <= split_date]
    test_df = df[df.index.get_level_values(0) > split_date]
    return train_df, test_df


class GetData:
    def __init__(self, factor_names, daily_path, startdate, enddate):
        self.factor_names = factor_names
        self.daily_path = daily_path
        self.startdate = startdate
        self.enddate = enddate

    def get_return_data(self):
        """
        获取 date[i] 与 date[i+1] 的return
        :return: dataframe -> index='date/code', columns='return'
        """
        with open(self.daily_path, 'rb') as file:
            ori_data = pickle.load(file)
        dates = [date for date in ori_data.keys() if (int(date) >= self.startdate) * (int(date) <= self.enddate)]

        dict_daily_return = {}
        # dict_daily_return[dates[0]] = (ori_data[dates[0]]['close'] / ori_data[dates[0]]['open']) - 1
        for i in range(0, len(dates)-1):
            data1 = ori_data[dates[i]]
            data = ori_data[dates[i+1]]
            dict_daily_return[dates[i]] = (data['close'] / data1['close']) - 1

        df_daily_return = pd.DataFrame(dict_daily_return).T

        # 对收益截面标准化
        row_mean = df_daily_return.mean(axis=1)
        row_std = df_daily_return.std(axis=1)
        df_daily_return = df_daily_return.apply(lambda x: (x - row_mean[x.name]) / row_std[x.name], axis=1)

        return_stacked = df_daily_return.stack(dropna=False).reset_index()
        return_stacked.columns = ['date', 'code', 'return']
        return_stacked.set_index(['date', 'code'], inplace=True)

        return return_stacked

    def get_factor_data(self, lag_day):
        """
        获取 date[i], date[i-1]... (根据lag_day决定) 的因子值
        :param lag_day: 取过去多少天的数据进行训练
        :return: dataframe -> index='date/code', columns=[factor_1_lag_1, factor_1_lag_2, ...]
        """
        factor_dfs = []
        multiindex_list = []
        for name in self.factor_names:
            path = os.path.join(folder_path, name, 'combination', paths[name])
            with open(path, 'rb') as file:
                ori_data = pickle.load(file)
            data = {date: value for date, value in ori_data.items() if str(self.startdate) <= date <= str(self.enddate)}

            # 获取多重索引
            multiindex = get_multiindex(data)
            multiindex_list.append(list(multiindex))

            # if iffill:
            #     # 对涨跌停的股票做一个填充
            #     data = cal_tool.filling_nan('mean')
            df = pd.DataFrame(data).T  # 此时有nan的值, 可能是因为st股或suspend的部分, 也或者是退市/还未上市的股票
            factor_dfs.append(df)

        lagged_features = pd.DataFrame()
        for idx, factor_df in enumerate(factor_dfs, start=1):
            for lag in range(0, lag_day + 1):
                lagged_feature = factor_df.shift(lag)
                lagged_stacked_feature = lagged_feature.stack()
                lagged_features[f'factor{idx}_lag_{lag}'] = lagged_stacked_feature

        # 找到feature中索引的交集
        common_multiindex_set = [set(multiindex_list_) for multiindex_list_ in multiindex_list]
        common_multiindex = list(reduce(lambda x, y: x & y, common_multiindex_set))
        common_multiindex = sorted(common_multiindex, key=lambda x: x[0])
        return lagged_features, common_multiindex

    def combine_data(self, feature_data, return_data):
        feature_data.index.names = ['date', 'code']
        return_data.index.names = ['date', 'code']
        df = pd.merge(feature_data, return_data, left_index=True, right_index=True, how='inner')

        df_cleaned = df.dropna()
        return df_cleaned


if __name__ == '__main__':
    startdate = int(20220101)
    enddate = int(20221231)

    gd = GetData(factor_names, daily_stock_path, startdate, enddate)

    return_data = gd.get_return_data()

    lag_day = 5
    iffill = False

    feature_data, common_multiindex = gd.get_factor_data(lag_day)

    df = gd.combine_data(feature_data, return_data)
