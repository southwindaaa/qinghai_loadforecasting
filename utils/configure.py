import json
import os


def load_config(file_path="Configure.json"):
    # 读取 JSON 文件
    assert os.path.exists(file_path),'Configure file not found in path: {}'.format(os.getcwd())
    with open(file_path, "r") as f:
        config = json.load(f)
    return config
