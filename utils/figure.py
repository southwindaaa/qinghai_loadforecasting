import logging
import warnings

from matplotlib import pyplot as plt
import os
# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
warnings.filterwarnings('ignore')


def draw(args, real, pred, net_id, save_path, mae_sample, mse_sample, mape_sample, iteration):
    # 创建一个绘图
    plt.figure(figsize=(12, 6))

    # 绘制 preds 和 trues 的曲线
    plt.plot(pred, label='Predictions', alpha=0.7)
    plt.plot(real, label='True Values', alpha=0.7)
    # print('Predictions vs True Values feature: '+ str(feat_ids[random_index,0]))
    plt.title('net_id:{}'.format(net_id))
    plt.xlabel('time_steps')
    plt.ylabel('Values')
    plt.legend(title=f'MAE: {mae_sample:.4f}\nMSE: {mse_sample:.4f}\nMAPE: {mape_sample:.4f}')

    path = os.path.join(save_path, args.model_comment, str(net_id))
    os.makedirs(path, exist_ok=True)
    # 保存图像
    logging.info(f'netid:{net_id},iter:{iteration}' + ' saved')
    plt.savefig(os.path.join(path, str(iteration) + '.jpg'))
