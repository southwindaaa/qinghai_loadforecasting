import os
import time

import numpy as np
import torch
from tqdm import tqdm

from utils.tools import adjust_learning_rate


def training(args,epoch,train_loader,model_optim,device, model,criterion,scheduler,scaler,save_path):
    model.train()
    iter_count = 0
    train_loss = []
    epoch_time = time.time()
    iter_count += 1
    # 创建 tqdm 进度条，设置总批次数
    with tqdm(total=args.train_epochs, desc=f"Training", unit="batch") as pbar:
        for i, batch in enumerate(train_loader):
            model_optim.zero_grad()
            batch_x = batch[0].float().to(device)
            batch_y = batch[1].float().to(device)
            batch_x_mark = batch[2].float().to(device)
            batch_y_mark = batch[3].float().to(device)
            net_id = batch[4].float().to(device)
            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(
                device)
            dec_inp = torch.cat([batch_x[:, :args.pred_len, -1:], dec_inp], dim=1).float().to(
                device)
            # 模型训练
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
            else:
                outputs = model(batch_x, None, dec_inp, None)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

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

            # 更新 tqdm 进度条，并显示自定义指标
            pbar.set_postfix({"Loss": f"{loss:.4f}", "Mean Loss": f"{np.average(train_loss):.4f}"})
            pbar.update(1)
            checkpoint_path = save_path + '/checkpoint-{}'.format(epoch)
            os.makedirs(checkpoint_path,exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path + '/model_weights.pth')
    return train_loss
