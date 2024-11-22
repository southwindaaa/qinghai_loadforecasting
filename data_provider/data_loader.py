import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler,LabelEncoder
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
import warnings
from collections import defaultdict
import pickle
import random
import torch
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')

def get_cols(other_id):
    if other_id is None:
        cols = ['netid','ymd']
    else:
        cols = ['netid']+[other_id]+['ymd']
    for hour in range(24):
        for minute in [0, 15, 30, 45]:
            cols.append(f"t{hour:02d}{minute:02d}")
    # print(time_strings)
    return cols

# 数据预处理和 CSV 读取的 class
class QinghaiLoadData(Dataset):
    def __init__(self, flag='train',root_path=None, 
                 size=None, data_path=None,
                 other_id=None,
                 scale=0,args=None):
        if size is None:
            self.seq_len = 672
            self.pred_len = 336
        else:
            self.seq_len = size[0]
            self.pred_len = size[2]
        if other_id==None:
            self.other_id_flag = 0
        else:
            self.other_id_flag = 1
        self.scale = scale
        self.set_map = {'train':0,'val':1}
        self.flag = self.set_map[flag]
        self.train_date = args.train_date
        # 读取 CSV 文件
        df = pd.read_csv(root_path+data_path)
        cols = get_cols(other_id)
        df = df[cols]
        
        if other_id is not None:
            df.columns = ['net_id', 'other_id', 'ymd'] + [str(i) for i in range(1, 97)]
        else:
            df.columns = ['net_id', 'ymd'] + [str(i) for i in range(1, 97)]
        print(df.columns)
        
        df['ymd'] = pd.to_datetime(df['ymd'], format='%Y%m%d')
        if self.flag == 0: #训练集
            df = df[df['ymd']<=pd.to_datetime(self.train_date, format='%Y%m%d')]
        else: #验证集
            df = df[df['ymd']>pd.to_datetime(self.train_date, format='%Y%m%d')]
        print ('after_seperate_df.shape',df.shape)
        
        self.scale_label_encoder_netid = LabelEncoder()
        self.scale_label_encoder_netid.fit(df['net_id'])
        if self.other_id_flag==1:
            self.scale_label_encoder_otherid = LabelEncoder()
            df = df.sort_values(by=['net_id', 'other_id', 'ymd'])
            self.data = self._handle_missing_dates(df)
            self.scale_label_encoder_otherid.fit(df['other_id'])
        else:
            df = df.sort_values(by=['net_id', 'ymd'])
            self.data = self._handle_missing_dates_no_otherid(df)
            
        print ('self.data',len(self.data))
        print('\n')
        
        
        if self.scale:
            data_all = []
            for net_id, other_id, all_data in self.data:
                data_all.append(all_data)
            data_all = np.array(data_all)
            self.scaler = StandardScaler()
            self.scaler.fit(data_all.reshape(1, -1))
            self.data_new = []
            for net_id, other_id, all_data in self.data:
                self.data_new.append((net_id, other_id, self.scaler.transform(all_data.reshape(1, -1))[0]))
            self.data = self.data_new
        
        
    def _handle_missing_dates(self, df):
        """
        处理 'ymd' 缺失的情况，将每条数据根据前后完整的数据分割成多条数据。
        """
        data = []
        for (net_id, other_id), group in df.groupby(['net_id', 'other_id']):
            print(net_id,other_id)
            group_data = group.drop(columns=['net_id', 'other_id'])  # 删除不需要的列
            group_data = group_data.set_index('ymd')
            
            # 检查每一天是否缺失数据
            date_range = pd.date_range(group_data.index.min(), group_data.index.max(), freq='D')
            missing_dates = set(date_range) - set(group_data.index)
            print(missing_dates)
            
            # 如果有缺失日期，则把缺失前后数据分开
            # print(len(group_data),len(group_data.values.flatten()))
            start_idx = 0
            last_idx = 0
            for current_idx in range(1, len(group_data)):
                # 如果当前日期和前一天的日期之间有缺失
                if (group_data.index[current_idx] - group_data.index[last_idx]).days > 1:
                    # 将两段数据分开
                    # print ('split',group_data.index[current_idx],group_data.index[last_idx])
                    # print ('netid',net_id,'other_id',other_id,
                    #     'last_idx',last_idx,
                    #     'shape:',group_data.iloc[start_idx:current_idx].values.flatten().shape)
                    data.append((net_id, other_id, group_data.iloc[start_idx:current_idx].values.flatten()))
                    start_idx = current_idx
                last_idx = current_idx
                
            # 最后一个区间的数据
            print ('netid',net_id,'other_id',other_id,
                   'last_idx',last_idx,
                   'shape:',group_data.iloc[start_idx:].values.flatten().shape)
            data.append((net_id, other_id, group_data.iloc[start_idx:].values.flatten()))
        
        return data
    
    def _handle_missing_dates_no_otherid(self, df):
        """
        处理 'ymd' 缺失的情况，将每条数据根据前后完整的数据分割成多条数据。
        """
        data = []
        for (net_id), group in df.groupby(['net_id']):
            group_data = group.drop(columns=['net_id'])  # 删除不需要的列
            group_data = group_data.set_index('ymd')
            
            # 检查每一天是否缺失数据
            date_range = pd.date_range(group_data.index.min(), group_data.index.max(), freq='D')
            missing_dates = set(date_range) - set(group_data.index)
            
            # 如果有缺失日期，则把缺失前后数据分开
            # print(len(group_data),len(group_data.values.flatten()))
            start_idx = 0
            last_idx = 0
            for current_idx in range(1, len(group_data)):
                # 如果当前日期和前一天的日期之间有缺失
                if (group_data.index[current_idx] - group_data.index[last_idx]).days > 1:
                    # 将两段数据分开
                    # print ('split',group_data.index[current_idx],group_data.index[last_idx])
                    # print ('netid',net_id,'other_id',other_id,
                    #     'last_idx',last_idx,
                    #     'shape:',group_data.iloc[start_idx:current_idx].values.flatten().shape)
                    data.append((net_id, group_data.iloc[start_idx:current_idx].values.flatten()))
                    start_idx = current_idx
                last_idx = current_idx
                
            # 最后一个区间的数据
            # print ('netid',net_id,'other_id',other_id,
            #        'last_idx',last_idx,
            #        'shape:',group_data.iloc[start_idx:].values.flatten().shape)
            data.append((net_id, group_data.iloc[start_idx:].values.flatten()))
        # print(len(data))
        # net_id, group_data = data[1]
        # print(len(group_data))
        
        return data
    
    def __len__(self):
        # 返回所有 net_id 和 other_id 组合的总数量 * 1000
        return len(self.data)*1000
    
    def __getitem__(self, idx):
        # 获取数据
        if self.other_id_flag:
            # print(idx%len(self.data))
            net_id, other_id, all_data = self.data[idx%len(self.data)]
            other_id = self.scale_label_encoder_otherid.transform([other_id])[0]
        else:
            # print(idx%len(self.data))
            net_id, all_data = self.data[idx%len(self.data)]
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
        if self.other_id_flag:
            return x_tensor, y_tensor, x_tensor, y_tensor,net_id, other_id
        else:
            return x_tensor, y_tensor, x_tensor, y_tensor,net_id
        
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
