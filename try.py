import pandas as pd
 
df = pd.read_csv('/root/qinghai_loadforecasting/data/load_data.csv', header=None)
time_strings = []
for hour in range(24):
    for minute in [0, 15, 30, 45]:
        time_strings.append(f"t{hour:02d}{minute:02d}")
 
# 指定列名
column_names = ['netid', 'otherid','ymd']+ time_strings  # 根据你的CSV文件实际列数调整
print(column_names)
# 给DataFrame添加列名
df.columns = column_names
 
# # 将带有列名的DataFrame写入新的CSV文件
# df.to_csv('/root/qinghai_loadforecasting/data/load_data_2.csv', index=False)

