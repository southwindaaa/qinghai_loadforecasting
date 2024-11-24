import pandas as pd
import os
import argparse
# 示例数据


# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description="Process some integers.")

# 添加命令行参数
parser.add_argument('data_path', type=str, default='load_data.csv'
                    , help='data file save path')
parser.add_argument('save_path', type=str, default='data.csv'
                    , help='processed data file save path')
# 解析命令行参数
args = parser.parse_args()

data = pd.read_csv(args.data_path)
# 使用正则表达式匹配列名
pattern = r'^t\d{4}$'  # 匹配以 t 开头，后面跟四个数字，且只有这个模式的列名
# 提取列名符合正则表达式的列
df_extracted = data.filter(regex=pattern)
# 定义要去除的列名
drop_columns = ['t2400']
df_extracted = df_extracted.drop(drop_columns,axis=1)
# 定义要额外添加的列名
extra_columns = ['netid','otherid','ymd']
output = pd.concat([data[extra_columns],df_extracted],axis=1)
output.to_csv(args.save_path,index=False)
print(f"数据集输出完毕，保存在{args.save_path},shape:{output.shape}")
