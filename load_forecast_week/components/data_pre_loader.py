import os

import joblib
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
import warnings
from collections import defaultdict
import pickle
import random
import torch
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


# 数据预处理和 CSV 读取的 class
class MyPreDataLoader(Dataset):
    def __init__(self, root_path, data_path, seq_len, pred_len, scaler,logging):
        # 数据路径
        self.root_path = root_path
        self.data_path = data_path
        # 序列长度
        self.seq_len = 672 if seq_len is None else seq_len
        # 预测长度
        self.pred_len = 336 if pred_len is None else pred_len
        # 是否标准化
        assert scaler, 'scaler 未导入'
        self.scaler = scaler
        self.logging = logging

        # 读取 CSV 文件
        self.logging.info(f'设定初始化完成，开始读取数据{os.path.join(self.root_path, self.data_path)}')
        df = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # 分别提取属性项和数据项
        pattern = r'^t\d{4}$'  # 匹配以 t 开头，后面跟四个数字，且只有这个模式的列名
        data = df.filter(regex=pattern)  # 提取数据项
        attr = df[df.columns[~df.columns.str.match(pattern)]]  # 提取属性项
        self.logging.info(f'数据读取完成，数据维度为{df.shape}')
        self.logging.info(f'属性名称为{attr.columns}')

        # 行内缺失值处理
        if data.isnull().sum().sum() > 0:
            data = self.fill_na(data)

        # 数据项进行标准化
        self.logging.info(f'开始标准化')
        data = self.scaler.transform(data)
        self.logging.info(f'标准化完成')
        df = pd.concat([attr, pd.DataFrame(data)], axis=1)

        self.scale_label_encoder_netid = LabelEncoder()
        self.scale_label_encoder_netid.fit(df['net_id'])

        df['ymd'] = pd.to_datetime(df['ymd'], format='%Y%m%d')
        self.data = self.handle_missing_dates(df)
        # self.logging.info('数据初始化完成，概要信息为：')
        # print(self.data)

    def fill_na(self, data, miss_threshold=0.25):
        # 输出数据集的缺失统计信息
        na_sum = data.isna().sum().sum()
        na_ratio = data.isna().mean().mean()
        self.logging.info(f'缺失值数量：{na_sum},缺失率：{na_ratio}')
        self.logging.debug('详细缺失信息：')
        value_na = data.isna().sum(axis=1)
        value_na = pd.concat([value_na, value_na / data.size], axis=1)
        value_na.columns = ['count', 'ratio']
        self.logging.debug(value_na)

        # 3. 如果缺失的值超过当前行的25%，删除该行
        df = data[data.isna().mean(axis=1) <= miss_threshold]
        # 1. 如果缺失值前后都有值，取前后值的平均值
        df = df.interpolate(method='linear', axis=1)
        # 2. 如果前面或者后面的值也缺失，取当前行的平均值填补
        df = df.apply(lambda row: row.fillna(row.mean()), axis=1)
        if df.isna().sum().sum() > 0:
            self.logging.warning('填充后仍有缺失')
        self.logging.info('数据填充完成')
        return df

    def handle_missing_dates(self, df):
        """
        处理 'ymd' 缺失的情况，将每条数据根据前后完整的数据分割成多条数据。
        """
        data = []
        for net_id, group in df.groupby('net_id'):
            group_data = group.drop(columns=['net_id'])  # 删除不需要的列
            group_data = group_data.set_index('ymd')
            data.append((net_id, group_data.values.flatten()))
        return data

    def __len__(self):
        # 返回所有 net_id 和 other_id 组合的总数量 * 1000
        return len(self.data) * 1000

    def __getitem__(self, idx):
        net_id, all_data = self.data[idx % len(self.data)]
        net_id = self.scale_label_encoder_netid.transform([net_id])[0]

        # 随机选择一个起始点
        max_start_idx = len(all_data) - self.seq_len - self.pred_len
        # print(len(all_data),self.seq_len,self.pred_len,max_start_idx)
        start_idx = random.randint(0, max_start_idx)
        # 输入序列
        x = all_data[start_idx:start_idx + self.seq_len]
        # 预测序列
        y = all_data[start_idx + self.seq_len:start_idx + self.seq_len + self.pred_len]
        # 转换为 tensor
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        return x_tensor, y_tensor, x_tensor, y_tensor, net_id

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def inverse_label_encoder(self, data):
        return self.scale_label_encoder_netid.inverse_transform(data)

# if __name__ == '__main__':
#     QinghaiLoadData(flag='train', root_path='../data/test', data_path='96load.csv', size=[336, 336, 96], scale=True)
