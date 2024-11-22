import os  
import matplotlib.pyplot as plt  
import numpy as np  
from .metrics import metric
import pickle

def draw_comparision(data_path,ii):
    # print(data_path.split('/'))
    with open(data_path, 'rb') as file:
        data = pickle.load(file)
    print(data['data_x'].shape,data['data_y'].shape,data['pred_y'].shape)
    
    true_sample = data['data_y'][-1]
    pred_sample = data['pred_y'][-1]
    # print(true_sample)
    # print(pred_sample)

    mae_sample, mse_sample, rmse_sample, mape_sample, mspe_sample = metric(pred_sample, true_sample)

    # 创建一个绘图
    plt.figure(figsize=(12, 6))


    # 绘制 preds 和 trues 的曲线
    plt.plot(pred_sample, label='Predictions', alpha=0.7)
    plt.plot(true_sample, label='True Values', alpha=0.7)
    # print('Predictions vs True Values feature: '+ str(feat_ids[random_index,0]))
    plt.title('Predictions vs True Values feature: ')
    plt.xlabel('Samples')
    plt.ylabel('Values')
    plt.legend(title=f'MAE: {mae_sample:.4f}\nMSE: {mse_sample:.4f}\nMAPE: {mape_sample:.4f}')

    result_folder = './results/predict_images/9net/'
    if not os.path.exists(result_folder):        
        os.makedirs(result_folder)
    # 保存图像
    print(result_folder+data_path.split('/')[2]+'_'+str(ii)+'.jpg')
    plt.savefig(result_folder+data_path.split('/')[2]+'_'+str(ii)+'.jpg')

    return mae_sample,mse_sample,mape_sample