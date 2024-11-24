import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from data_provider.data_pre_loader import MyPreDataLoader
from models import DLinear
from utils.metrics import metric


def predict(args, device, model, predict_loader):
    print(len(predict_loader))
    model.eval()
    model.to(device)
    data_x = []
    data_y = []
    pred_y = []
    net_ids = []
    other_ids = []

    with torch.no_grad():
        for i, batch in tqdm(enumerate(predict_loader)):
            batch_x = batch[0].float().to(device)
            batch_y = batch[1].float().to(device)
            batch_x_mark = batch[2].float().to(device)
            batch_y_mark = batch[3].float().to(device)
            net_id = batch[4].float().to(device)
            if args.other_id == 1:
                other_id = batch[5].float().to(device)
            else:
                other_id = None


            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(
                device)
            # print ('dec_inp',dec_inp)
            dec_inp = torch.cat([batch_x[:, :args.pred_len, :], dec_inp], dim=1).float().to(
                device)
            # encoder - decoder

            batch_x_mark = None
            batch_y_mark = None
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            # outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))

            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :].to(device)

            pred = outputs.detach()
            true = batch_y.detach()

            data_x.append(batch_x.detach().cpu().numpy())
            data_y.append(true.detach().cpu().numpy())
            pred_y.append(pred.detach().cpu().numpy())
            net_ids.append(net_id.detach().cpu().numpy())
            if args.other_id == 1:
                other_ids.append(other_id.detach().cpu().numpy())

    data_x = np.concatenate(data_x, axis=0)
    data_y = np.concatenate(data_y, axis=0)
    pred_y = np.concatenate(pred_y, axis=0)
    net_ids = np.concatenate(net_ids, axis=0)
    r2 = r2_score(data_y.reshape(-1, 1), pred_y.reshape(-1, 1))

    predict_data = {}
    predict_data['data_x'] = data_x
    predict_data['data_y'] = data_y
    predict_data['pred_y'] = pred_y
    predict_data['net_ids'] = net_ids
    if args.other_id == 1:
        predict_data['other_ids'] = other_ids
    return predict_data


def draw(args, real, pred, net_id, save_path, mae_sample, mse_sample, mape_sample, i):
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

    path = os.path.join(save_path, args.model_comment, net_id)
    os.makedirs(path, exist_ok=True)
    # 保存图像
    print(f'netid:{net_id},iter:{i}' + ' saved')
    plt.savefig(os.path.join(path, str(i) + '.jpg'))


# 从 JSON 文件读取配置参数
with open("Configure.json", "r") as f:
    args_dict = json.load(f)

args = argparse.Namespace(**args_dict['params'])
device = torch.device('cuda:0')
model = DLinear.Model(args)
model.load_state_dict(
    torch.load("checkpoints/long_term_forecast_DLinear_qinghaidata_test_GPT2_d_ff_32-96points/model_weights.pth"))
scaler = joblib.load('scaler/load_data/20241122_1307scaler.pkl')
# 进行预测
preLoader = MyPreDataLoader('data', 'load_data.csv', 336, 96, scaler)
data_loader = DataLoader(
    preLoader,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    drop_last=True)
predictions = predict(args, device, model, data_loader)
dataY = preLoader.inverse_transform(predictions['data_y'].squeeze())
predY = preLoader.inverse_transform(predictions['pred_y'].squeeze())
netIds = preLoader.inverse_label_encoder(predictions['net_ids'].astype(np.int32))
for i, net_id in enumerate(netIds):
    true_sample = dataY[i, :]
    pred_sample = predY[i, :]
    mae_sample, mse_sample, rmse_sample, mape_sample, mspe_sample = metric(pred_sample, true_sample)
    draw(args, true_sample, pred_sample, net_id, 'result', mae_sample, mse_sample, mape_sample, i)
print(1)
