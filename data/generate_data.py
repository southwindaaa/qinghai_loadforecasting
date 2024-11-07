import random
import pandas as pd
from datetime import datetime, timedelta

# 随机生成net_id和other_id
net_ids = [f"net_{i}" for i in range(1, 4)]
other_ids = [f"other_{i}" for i in range(1, 4)]

# 生成日期范围
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 1, 30)
date_list = pd.date_range(start_date, end_date).strftime('%Y%m%d').tolist()

# 创建数据
data = []

for date in date_list:
    for net_id in net_ids:
        for other_id in other_ids:
            # 随机生成96个点的负荷数据（这里生成的是0到100之间的随机数）
            loads = [random.randint(0, 100) for _ in range(96)]
            temp_data = [net_id, other_id, date]
            temp_data.extend(loads)
            data.append(temp_data)

# 创建DataFrame
df = pd.DataFrame(data, columns=["net_id", "other_id", "ymd", *range(1, 97)])

# 保存为CSV文件
df.to_csv("load_data.csv", index=False)

print("CSV文件已生成：load_data.csv")