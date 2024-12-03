import logging
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
                              end_date='2024-01-30',drop_date=False):
    # 随机生成net_id和other_id
    net_ids = [f"net_{i}" for i in range(max_net_id)]
    # 生成日期范围
    date_list = pd.date_range(start_date, end_date).strftime('%Y%m%d').tolist()
    # 创建数据
    areaId = []
    for date in date_list:
        for net_id in net_ids:
            areaId.append({'net_id': net_id, 'date': date})
    areaId = pd.DataFrame(areaId)
    # 随机丢弃行
    if drop_date:
        rows_to_drop = np.random.choice(areaId.index, size=int(areaId.shape[0] * drop_ratio), replace=False)
        areaId = areaId.drop(index=rows_to_drop).reset_index(drop=True)
    # 天级别数据随机缺失
    sequences = [np.random.randint(min, high, size=length) for _ in range(areaId.shape[0])]
    for miss_row in generate_missing_pos(areaId.shape[0], miss_row_ratio):
        sequences[miss_row] = generate_sequence_with_missing(length, min, high, miss_ratio)
    # 数据格式化
    data = pd.concat([areaId, pd.DataFrame(sequences)], axis=1)
    time_strings = ["t" + t.strftime("%H%M") for t in pd.date_range(start="00:00", end="23:45", freq="15min")]
    data.columns = ["net_id", "ymd"] + time_strings
    return data


def generate_data_day(start_date='2024-01-01', end_date='2024-01-30',
                      max_net_id=4, drop_ratio=0.1,
                      miss_row_ratio=0.1, miss_ratio=0.2,
                      min=1, high=100, length=96, exit_other_id=False
                      ):
    if exit_other_id:
        return None
    else:
        return generate_data_no_other_id(start_date=start_date, end_date=end_date,
                                         max_net_id=max_net_id, drop_ratio=drop_ratio,
                                         miss_row_ratio=miss_row_ratio, miss_ratio=miss_ratio,
                                         min=min, high=high, length=length)


def fill_na(self, data, miss_threshold=0.25):
    # 输出数据集的缺失统计信息
    na_sum = data.isna().sum().sum()
    na_ratio = data.isna().mean().mean()
    logging.info(f'缺失值数量：{na_sum},缺失率：{na_ratio}')
    logging.debug('详细缺失信息：')
    value_na = data.isna().sum(axis=1)
    value_na = pd.concat([value_na, value_na / data.size], axis=1)
    value_na.columns = ['count', 'ratio']
    logging.debug(value_na)

    # 3. 如果缺失的值超过当前行的25%，删除该行
    df = data[data.isna().mean(axis=1) <= miss_threshold]
    # 1. 如果缺失值前后都有值，取前后值的平均值
    df = df.interpolate(method='linear', axis=1)
    # 2. 如果前面或者后面的值也缺失，取当前行的平均值填补
    df = df.apply(lambda row: row.fillna(row.mean()), axis=1)
    if df.isna().sum().sum() > 0:
        logging.warning('填充后仍有缺失')
    logging.info('数据填充完成')
    return df


import pandas as pd


def get_date_week_range(year, week):
    # 使用 pandas 的 Timestamp 和 isocalendar
    first_day = pd.Timestamp(f'{year}-01-01')

    # 找到这一年的第一个周一（ISO 的第一周）
    start_of_week_1 = first_day + pd.offsets.Week(weekday=0) - pd.offsets.Week()

    # 计算目标周的开始日期
    start_date = start_of_week_1 + pd.offsets.Week(week - 1)
    end_date = start_date + pd.offsets.Day(6)  # 周结束日期为起始日期+6天

    return start_date, end_date


def get_date_month_range(year, week):
    # 使用 pandas 的 Timestamp 和 isocalendar
    first_day = pd.Timestamp(f'{year}-01-01')

    # 找到这一年的第一个周一（ISO 的第一周）
    start_of_week_1 = first_day + pd.offsets.Mon- pd.offsets.Week()

    # 计算目标周的开始日期
    start_date = start_of_week_1 + pd.offsets.Week(week - 1)
    end_date = start_date + pd.offsets.Day(6)  # 周结束日期为起始日期+6天

    return start_date, end_date

def generate_data_offset(data_path,offset_days=7):
    df = pd.read_csv(data_path)
    pattern = r'^t\d{4}$'  # 匹配以 t 开头，后面跟四个数字，且只有这个模式的列名
    data = df.filter(regex=pattern)  # 提取数据项
    attr = df[df.columns[~df.columns.str.match(pattern)]]
    attr = attr.copy()
    attr['value'] = data.sum(axis=1)  # 计算每行的最大值
    attr['ymd'] = pd.to_datetime(attr['ymd'], format='%Y%m%d')
    start_date = attr['ymd'].min()
    attr.set_index('ymd', inplace=True)
    colums = ['net_id', 'ymd'] + [str(f"{t:04}") for t in range(1, 8)]
    data = []
    for netid, item in attr.groupby('net_id'):
        # 提取偏移量内的数据
        end_date = start_date + pd.Timedelta(days=offset_days - 1)
        subset = item.loc[start_date:end_date]
        info = [netid, start_date.strftime('%Y%m%d')] + subset['value'].values.tolist()
        data.append({key: value for key, value in zip(colums, info)})
    return pd.DataFrame(data).dropna()

def generate_data_week(data_path):
    df = pd.read_csv(data_path)
    pattern = r'^t\d{4}$'  # 匹配以 t 开头，后面跟四个数字，且只有这个模式的列名
    data = df.filter(regex=pattern)  # 提取数据项
    attr = df[df.columns[~df.columns.str.match(pattern)]]
    attr = attr.copy()
    attr['value'] = data.sum(axis=1)  # 计算每行的最大值
    attr['ymd'] = pd.to_datetime(attr['ymd'], format='%Y%m%d')
    # 提取每条数据所属的 ISO 年和周
    attr['year'] = attr['ymd'].dt.isocalendar().year  # ISO 年份
    attr['week'] = attr['ymd'].dt.isocalendar().week  # ISO 周数
    data = []
    colums = ['net_id', 'ymd'] + ["t000" + str(t) for t in range(1, 8)]
    for netid, item in attr.groupby('net_id'):
        for (year, week), group in item.groupby(['year', 'week']):
            # 创建一个完整的日期范围
            start_date, end_date = get_date_week_range(year, week)
            full_week = pd.date_range(start=start_date, end=end_date, freq='D')
            # 重新设置 DataFrame，按完整日期范围补全数据
            df_full = pd.DataFrame({'ymd': full_week})
            df_full = df_full.merge(item, on='ymd', how='left')
            info = [netid, start_date.strftime('%Y%m%d')] + df_full['value'].values.tolist()
            data.append({key: value for key, value in zip(colums, info)})
    return pd.DataFrame(data).dropna()



def generate_data_month(data_path):
    df = pd.read_csv(data_path)
    pattern = r'^t\d{4}$'  # 匹配以 t 开头，后面跟四个数字，且只有这个模式的列名
    data = df.filter(regex=pattern)  # 提取数据项
    attr = df[df.columns[~df.columns.str.match(pattern)]]
    attr = attr.copy()
    attr['value'] = data.max(axis=1)  # 计算每行的最大值
    attr['ymd'] = pd.to_datetime(attr['ymd'], format='%Y%m%d')
    # 提取每条数据所属的 ISO 年和月
    attr['year'] = attr['ymd'].dt.isocalendar().year  # ISO 年份
    attr['month'] = attr['ymd'].dt.isocalendar().month  # ISO 月份
    data = []
    colums = ['net_id', 'ymd'] + [t for t in range(1, 31)]
    for netid, item in attr.groupby('net_id'):
        for (year, month), group in item.groupby(['year', 'month']):
            t_value = group.sort_values('weekday')['value'].values.tolist()
            info = [netid, f"{year}-{week}"] + t_value
            data.append({key: value for key, value in zip(colums, info)})
    data = pd.DataFrame(data)


if __name__ == '__main__':
    outputPath = '../load_forecast_week/data'
    fileName = 'data.csv'
    os.makedirs(outputPath, exist_ok=True)
    # data = generate_data_day(start_date='2020-01-01',end_date='2024-01-01')  # 生成日数据
    data = generate_data_offset('../load_forecast_day96/data/data.csv')  # 在日数据的基础上生成周数据
    data.to_csv(os.path.join(outputPath, fileName), index=False)
    print(f'{fileName}文件已生成，存放在{outputPath}目录下\n')
    print(f'shape: {data.shape},miss_rate:{data.isna().mean().mean()}')
