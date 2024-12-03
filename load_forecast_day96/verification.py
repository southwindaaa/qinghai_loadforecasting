import argparse
import json
import logging
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from components.data_pre_loader import MyPreDataLoader
from models import DLinear
from utils.figure import draw
from utils.log import setup_logger
from utils.metrics import metric
from utils.configure import load_config

warnings.filterwarnings('ignore')


def predict(args, device, model, predict_loader):
    model.eval()
    model.to(device)
    data_x = []
    data_y = []
    pred_y = []
    net_ids = []
    other_ids = []
    mse_total_loss = []
    mae_total_loss = []

    with torch.no_grad():
        with tqdm(total=args.train_epochs, desc=f"Testing", unit="batch") as pbar:
            for i, batch in enumerate(predict_loader):
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
                # 更新 tqdm 进度条，并显示自定义指标
                pbar.update(1)

    data_x = np.concatenate(data_x, axis=0)
    data_y = np.concatenate(data_y, axis=0)
    pred_y = np.concatenate(pred_y, axis=0)
    net_ids = np.concatenate(net_ids, axis=0)

    predict_data = {}
    predict_data['data_x'] = data_x
    predict_data['data_y'] = data_y
    predict_data['pred_y'] = pred_y
    predict_data['net_ids'] = net_ids
    return predict_data


# 从 JSON 文件读取配置参数
config = load_config()
args = argparse.Namespace(**config['train'])
infer = config['infer']
logging = setup_logger(config)
# 加载配置文件
device = torch.device('cuda:0')
model = DLinear.Model(args)
model.load_state_dict(torch.load(os.path.join(infer['model_path'], infer['comment'], infer['weight_name'])))
scaler = joblib.load(os.path.join(infer['model_path'], infer['comment'], infer['scaler_name']))
# 进行预测
preLoader = MyPreDataLoader(infer['data_path'], infer['data_name'], args.seq_len, args.pred_len, scaler,logging)
data_loader = DataLoader(preLoader, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                         drop_last=True)
predictions = predict(args, device, model, data_loader)
dataY = preLoader.inverse_transform(predictions['data_y'].squeeze())
predY = preLoader.inverse_transform(predictions['pred_y'].squeeze())
netIds = preLoader.inverse_label_encoder(predictions['net_ids'].astype(np.int32))
for i, net_id in enumerate(netIds):
    true_sample = dataY[i, :]
    pred_sample = predY[i, :]
    mae_sample, mse_sample, rmse_sample, mape_sample, mspe_sample = metric(predictions['pred_y'][i, :, :],
                                                                           predictions['data_y'][i, :, :])
    draw(args, true_sample, pred_sample, net_id, infer['visual_path'], mae_sample, mse_sample, mape_sample, i)
logging.info('Verification finished.')
