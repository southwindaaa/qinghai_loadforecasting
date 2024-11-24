import os
import random

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_missing_pos(length=96, missing_ratio=0.2):
    # 生成缺失值的位置
    num_missing = int(length * missing_ratio)  # 计算缺失值数量
    missing_indices = np.random.choice(length, num_missing, replace=False)  # 随机选择缺失位置
    return missing_indices

# 构建缺失值序列
def generate_sequence_with_missing(length=96, min=1, high=100, missing_ratio=0.2):
    sequence = np.random.randint(min, high, size=length)
    # 填充Nan
    sequence_with_missing = sequence.astype(float)  # 转换为浮点类型以支持 NaN
    sequence_with_missing[generate_missing_pos(length, missing_ratio)] = np.nan

    return sequence_with_missing


# 构建没有缺失值序列
def generate_sequence_no_missing(length=96, min=1, high=100):
    return np.random.randint(min, high, size=length)


def generate_data_no_other_id(max_net_id=4, drop_ratio=0.1, miss_row_ratio=0.1,
                              miss_ratio=0.2, start_date='2023-01-01',
                              min=1, high=100, length=96,
                              end_date='2024-01-30'):
    # 随机生成net_id和other_id
    net_ids = [f"net_{i}" for i in range(max_net_id)]
    # 生成日期范围
    date_list = pd.date_range(start_date, end_date).strftime('%Y%m%d').tolist()
    # 创建数据
    areaId = []
    for date in date_list:
        for net_id in net_ids:
            areaId.append({'netid': net_id, 'date': date})
    areaId = pd.DataFrame(areaId)
    # 随机丢弃行
    rows_to_drop = np.random.choice(areaId.index, size=int(areaId.shape[0] * drop_ratio), replace=False)
    areaId = areaId.drop(index=rows_to_drop).reset_index(drop=True)
    # 行随机缺失
    sequences = [np.random.randint(min, high, size=length) for _ in range(areaId.shape[0])]
    for miss_row in generate_missing_pos(areaId.shape[0], miss_row_ratio):
        sequences[miss_row] = generate_sequence_with_missing(length, min, high, miss_ratio)
    # 数据格式化
    data = pd.concat([areaId,pd.DataFrame(sequences)],axis=1)
    time_strings = ["t" + t.strftime("%H%M") for t in pd.date_range(start="00:00", end="23:45", freq="15min")]
    data.columns = ["netid", "ymd"]+time_strings
    return data


def generate_data_day(exit_other_id=False):
    if exit_other_id:
        return None
    else:
        return generate_data_no_other_id()

def generate_data_week(data_path):
    data = pd.read_csv(data_path)


def generate_data_month():
    return generate_data_no_other_id(exit_other_id=False)
if __name__ == '__main__':
    outputPath = '../data/test/'
    fileName = '96load.csv'
    os.makedirs(outputPath,exist_ok=True)
    # data = generate_data_day() # 生成日数据
    data = generate_data_week('../data/test/96load.csv') #在日数据的基础上生成周数据
    data.to_csv(os.path.join(outputPath,fileName), index=False)
    print(f'{fileName}文件已生成，存放在{outputPath}目录下\n')
    print(f'shape: {data.shape},miss_rate:{data.isna().mean().mean()}')
