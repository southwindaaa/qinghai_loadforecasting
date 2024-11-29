import argparse
import json
import warnings
from datetime import datetime

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import DLinear
import time
import random
import numpy as np
import os
import pickle

from utils.configure import load_config
from utils.log import setup_logger
from utils.painting import draw_comparision
from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content, vali_evaluation
from components.train import training
from components.vali import validating
from components.loader import dataLoader

warnings.filterwarnings('ignore')


# from accelerate import Accelerator, DeepSpeedPlugin
# from accelerate import DistributedDataParallelKwargs
# from models import Autoformer, DLinear, TimeLLM,FEDformer, TSMixer, PatchTST, iTransformer, LSTM

def set_model(args):
    if args.model == 'DLinear':
        model = DLinear.Model(args)
    # elif args.model == 'Autoformer':
    #     model = Autoformer.Model(args).float()
    # elif args.model == 'PatchTST':
    #     model = PatchTST.Model(args).float()
    # elif args.model == 'FEDformer':
    #     model = FEDformer.Model(args).float()
    # elif args.model == 'TSMixer':
    #     model = TSMixer.Model(args).float()
    # elif args.model == 'iTransformer':
    #     model = iTransformer.Model(args).float()
    # # elif args.model == 'TimeLLM':
    # #     if args.add_social:
    # #         model = TimeLLM_wSocial.Model(args).float()
    # #     else:
    # #         model = TimeLLM.Model(args).float()
    # elif args.model == 'LSTM':
    #     model = LSTM.Model(args).float()
    #     # model = TimeLLM.Model(args).float()
    else:
        raise NotImplementedError
    return model


def set_save_path(args):
    model_save_path = os.path.join(args.checkpoints, args.model_comment)
    os.makedirs(model_save_path, exist_ok=True)
    return model_save_path


def set_model_optim(model):
    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    trained_parameters_show = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trained_parameters_show.append((name, param))
    # 打印每个训练参数的名字
    for name, param in trained_parameters_show:
        logging.info(f"Parameter name: {name}")
    return trained_parameters


# 静态设置
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

config = load_config()
# 设置参数
args = argparse.Namespace(**config['train'])
save_path = set_save_path(args)
logging = setup_logger(config,save_path)
if args.is_training == 0 and args.checkpoint <= 0:
    raise ValueError("When --is_training is 0, --checkpoint must be greater than 0.")

for ii in range(args.itr):
    # 数据加载
    logging.info('数据加载开始')
    train_data = dataLoader(args, logging,save_path, 'train')
    vali_data = dataLoader(args, logging,save_path, 'val')
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                              drop_last=True)
    vali_loader = DataLoader(vali_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             drop_last=True)
    logging.info('数据加载完成\n')
    # 模型设置
    # 设置保存路径
    logging.info('开始设置模型参数及其他设置初始化')

    device = torch.device('cuda:0')
    model = set_model(args).float().to(device)
    train_steps = len(train_loader)
    early_stopping = EarlyStopping(accelerator=None, patience=args.patience)
    model_optim = optim.Adam(set_model_optim(model), lr=args.learning_rate)
    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
    else:
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=args.pct_start,
                                            epochs=args.train_epochs,
                                            max_lr=args.learning_rate)

    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    min_mse, min_mae, max_r2 = float('inf'), float('inf'), float('-inf')
    logging.info(f'模型参数及其他设置初始化完成,dataLoader长度为{train_steps}\n')
    logging.info('开始训练 。 。 。 。 。 。')
    start_time = time.time()
    for epoch in range(args.train_epochs):
        epoch_start_time = time.time()
        print(f"Epoch {epoch}/{args.train_epochs}")
        train_loss = training(args, epoch, train_loader, model_optim, device, model, criterion, scheduler, scaler,
                              save_path)
        vali_loss, vali_mae_loss, vali_r2 = validating(args, epoch, vali_loader, device, model, criterion, mae_metric,
                                                       save_path)

        min_mse = vali_loss if vali_loss < min_mse else min_mse
        min_mae = vali_mae_loss if vali_mae_loss < min_mae else min_mae

        if vali_r2 > max_r2:
            max_r2 = vali_r2 if vali_r2 > max_r2 else max_r2
            torch.save(model.state_dict(), os.path.join(save_path, 'model_weights.pth'))
        early_stopping(vali_loss, model, save_path)
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
                    logging.info("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                adjust_learning_rate(None, model_optim, scheduler, epoch + 1, args, printout=False)
        else:
            print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

    logging.info(f"运行结束,min_mse: {min_mse}, min_mae: {min_mae},max_r2: {max_r2}")

