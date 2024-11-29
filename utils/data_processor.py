import logging
import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_data_offset(data_path, offset_days=7):
    df = pd.read_csv(data_path)
    pattern = r'^t\d{4}$'  # 匹配以 t 开头，后面跟四个数字，且只有这个模式的列名
    data = df.filter(regex=pattern)  # 提取数据项
    attr = df[df.columns[~df.columns.str.match(pattern)]]
    attr = attr.copy()
    attr['value'] = data.sum(axis=1)  # 计算每行的最大值
    attr['ymd'] = pd.to_datetime(attr['ymd'], format='%Y%m%d')
    start = attr['ymd'].min()
    # 计算日期范围的天数（包括首尾）
    total_days = (attr['ymd'].max() - attr['ymd'].min()).days + 1
    attr.set_index('ymd', inplace=True)
    colums = ['net_id', 'ymd'] + [str(f"t{t:04}") for t in range(1, offset_days + 1)]
    data = []
    for netid, item in attr.groupby('net_id'):
        start_date = start
        # 提取偏移量内的数据
        for i in range(int(total_days / offset_days)):
            end_date = start_date + pd.Timedelta(days=offset_days - 1)
            subset = item.loc[start_date:end_date]
            info = [netid, start_date.strftime('%Y%m%d')] + subset['value'].values.tolist()
            data.append({key: value for key, value in zip(colums, info)})
            start_date = end_date + pd.Timedelta(days=1)
    return pd.DataFrame(data).dropna()


if __name__ == '__main__':
    outputPath = '../load_forecast_month/data'
    fileName = 'data.csv'
    os.makedirs(outputPath, exist_ok=True)
    # 以偏移量为7天生成数据
    # data = generate_data_offset('../load_forecast_day96/data/data.csv', 7)
    # 以偏移量为30天生成数据
    data = generate_data_offset('../load_forecast_day96/data/data.csv', 30)
    # 以偏移量为90天生成数据
    # data = generate_data_offset('../load_forecast_day96/data/data.csv',90)
    data.to_csv(os.path.join(outputPath, fileName), index=False)
    print(f'{fileName}文件已生成，存放在{outputPath}目录下\n')
    print(f'shape: {data.shape},miss_rate:{data.isna().mean().mean()}')
