import json
import yaml
import os

# 从当前目录下的 Configure.json 文件中读取配置信息
# 返回一个字典
def load_config(file_path="Configure.yaml"):
    # 读取 JSON 文件
    assert os.path.exists(file_path), 'Configure file not found in path: {}'.format(os.getcwd())
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    return config
