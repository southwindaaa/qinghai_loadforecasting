# 96点负荷预测模型
# 组件目录
    ├── data --存放数据集
    ├── components --存放模型组件
    ├  ├────   loader --数据加载组件
    ├  ├────   train --训练器
    ├  ├────   vali --验证器
    ├  ├────   data_pre_loader --推理数据加载器
    ├—— checkpoints --存放模型训练结果
    ├── result  ==存放模型推理可视化结果
    ├── Configure --存放模型配置文件
    ├── main.py --主程序入口
    ├── verification --模型推理代码
    └── .gitignore

# 不同电网96点负荷预测执行方式
    # 修改模型配置文件
    Configure.yaml
    # 执行模型训练
    python main.py
