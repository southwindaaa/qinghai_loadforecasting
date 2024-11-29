import logging
import os.path


def setup_logger(config, save_path=None):
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.INFO)

    # File handler
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(save_path, config['log']['log_name']))
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

    logger.info(f"项目{config['name']}开始运行,github版本号:{config['version']}")
    logger.info(f"子项目名称：{config['sub_name']}")
    logger.info(f"项目组：{config['group']}")
    logger.info(f"描述：{config['description']}\n")

    return logger
