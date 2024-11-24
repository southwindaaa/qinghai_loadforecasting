import argparse
import json

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from models import DLinear
from data_provider.data_factory import data_provider
import time
import random
import numpy as np
import os
import pickle
from utils.painting import draw_comparision
import logging
from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content, vali_evaluation





# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.basicConfig(level=logging.INFO)
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
device = torch.device('cuda:0')
# 从 JSON 文件读取配置参数
with open("Configure.json", "r") as f:
    args_dict = json.load(f)
args = argparse.Namespace(**args_dict['params'])

assert args.is_training == 0 and args.checkpoint <= 0, "When --is_training is 0, --checkpoint must be greater than 0."

# 存储路径设置
path = os.path.join(args.checkpoints, args.task_name + '-' + args.model_comment)
# 加载训练数据
train_data, train_loader = data_provider(args, 'train')
vali_data, vali_loader = data_provider(args, 'val')

model = DLinear.Model(args).float().to(device)
early_stopping = EarlyStopping(accelerator=None, patience=args.patience)
model_optim = optim.Adam([p for p in model.parameters() if p.requires_grad is True], lr=args.learning_rate)
if args.lradj == 'COS':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
else:
    scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                        steps_per_epoch=len(train_loader),
                                        pct_start=args.pct_start,
                                        epochs=args.train_epochs,
                                        max_lr=args.learning_rate)
if args.use_amp:
    scaler = torch.cuda.amp.GradScaler()
criterion = nn.MSELoss()
mae_metric = nn.L1Loss()

for epoch in range(args.train_epochs):
    iter_count = 0
    train_loss = []

    model.train()
    epoch_time = time.time()
    # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
    print(len(train_loader))
    for i, batch in tqdm(enumerate(train_loader)):
        iter_count += 1
        model_optim.zero_grad()
        batch_x = batch[0].float().to(device)
        batch_y = batch[1].float().to(device)
        batch_x_mark = batch[2].float().to(device)
        batch_y_mark = batch[3].float().to(device)
        net_id = batch[4].float().to(device)

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(
            device)
        dec_inp = torch.cat([batch_x[:, :args.pred_len, :], dec_inp], dim=1).float().to(
            device)

        if args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())
        else:
            outputs = model(batch_x, None, dec_inp, None)
            # f_dim = -1 if args.features == 'MS' else 0
            # outputs = outputs[:, -args.pred_len:, f_dim:]
            # batch_y = batch_y[:, -args.pred_len:, f_dim:]
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())
            # print ('outputs',outputs.shape,'batch_y',batch_y.shape)

        if (i + 1) % 100 == 0:
            print(
                "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
            speed = (time.time() - time_now) / iter_count
            left_time = speed * ((args.train_epochs - epoch) * len(train_loader) - i)
            print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
            iter_count = 0
            time_now = time.time()

        if args.use_amp:
            scaler.scale(loss).backward()
            scaler.step(model_optim)
            scaler.update()
        else:
            loss.backward()
            model_optim.step()

        if args.lradj == 'TST':
            adjust_learning_rate(None, model_optim, scheduler, epoch + 1, args, printout=False)
            scheduler.step()

    logging.info(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
    # 保存每轮的训练权重
    checkpoint_path = path + '/checkpoint-{}'.format(epoch)
    os.makedirs(checkpoint_path, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path + '/model_weights.pth')
    train_loss = np.average(train_loss)
    validation_data = vali_evaluation(args, device, model, vali_data, vali_loader, criterion, mae_metric)
    vali_loss = validation_data['total_loss']
    vali_mae_loss = validation_data['total_mae_loss']
    vali_r2 = validation_data['total_r2']

    if vali_loss < min_mse:
        min_mse = vali_loss
    if vali_mae_loss < min_mae:
        min_mae = vali_mae_loss
    if vali_r2 > max_r2:
        max_r2 = vali_r2
        torch.save(model.state_dict(), path + '/model_weights.pth')
    pickle.dump(validation_data, open(checkpoint_path + '/test_data.pkl', 'wb'))
    logging.info(
        "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f} MAE Loss: {4:.7f}, r2: {5:.7f}".format(
            epoch + 1, train_loss, vali_loss, vali_loss, vali_mae_loss, vali_r2))
    # accelerator.print(
    #     "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f}".format(
    #         epoch + 1, train_loss, vali_loss))
    early_stopping(vali_loss, model, path)
    if early_stopping.early_stop:
        print("Early stopping")
        break

    if args.lradj != 'TST':
        if args.lradj == 'COS':
            scheduler.step()
            print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
        else:
            if epoch == 0:
                args.learning_rate = model_optim.param_groups[0]['lr']
                print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            adjust_learning_rate(None, model_optim, scheduler, epoch + 1, args, printout=True)

    else:
        print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
